from typing import Optional, Tuple
from tqdm import tqdm
import numpy as np
import torch
from torch import nn
import os
import math
import matplotlib.pyplot as plt

import ase
from ase.io import read, write
from ase.cluster.wulff import wulff_construction

from env import OffLatticeKMC, ACTION_SPACES
from tools.calc import Calculator

K = 8.6173324e-05
TEMPERATURE_K = 473.15
R = 8.314
KCAL_2_EV = 96485 / 4.184

class RunKMC():
    def __init__(self, 
                 model_path:str = 'PdO',
                 calculate_method:str = 'LASP',
                 max_observaton_atoms:int = 400,
                 save_dir:Optional[str]=None,
                 save_every:Optional[int] = None,
                 ) -> None:
        self.initial_slab = self._generate_initial_slab()
        self.calculator = Calculator(model_path = model_path, calculate_method=calculate_method, temperature_K=TEMPERATURE_K)
        self.atoms, initial_energy, _ = self.calculator.to_calc(self.initial_slab, calc_type = 'opt')

        self.env = OffLatticeKMC(initial_slab=self.initial_slab, 
                                 calculator = self.calculator, 
                                 cutoff = 4.0, 
                                 )
        
        self.initial_energy = initial_energy + self.env.n_O2 * self.env.E_O2 + self.env.n_O3 * self.env.E_O3
        self.max_observation_atoms = max_observaton_atoms
        self.pd = nn.ZeroPad2d(padding = (0,0,0,self.max_observation_atoms-len(self.atoms.get_positions())))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.history_dir = os.path.join(save_dir, 'history')
        self.plot_dir = os.path.join(save_dir, 'plots')
        self.save_every = save_every
        if not os.path.exists(self.history_dir):
            os.makedirs(self.history_dir)
        if not os.path.exists(self.plot_dir):
            os.makedirs(self.plot_dir)

        results = ['energies', 'actions', 'structures', 'timesteps', 'barriers']
        self.history = {}
        for item in results:
            self.history[item] = []
        self.history['energies'] = [self.initial_energy]
        self.history['actions'] = [0]
        self.history['structures'] = [np.array(self.pd(torch.tensor(self.atoms.get_positions())).flatten())]
        self.history['timesteps'] = [0]
        self.history['real_time'] = [0.0]
        self.history['barriers'] = [0]
        self.history['adsorbates'] = [(self.env.n_O2, self.env.n_O3)]

    def run(self, atoms:ase.Atoms, num_episode: int) -> ase.Atoms:
        previous_atoms = atoms.copy()
        previous_energy = self.history['energies'][-1]
        # print(f"The previous energy is {previous_energy}")
        atoms, tmp_state_dict = self.env.step(atoms, previous_energy, num_episode)

        current_energy = tmp_state_dict['energy']
        barrier = tmp_state_dict['barrier']

        # print(f"The previous n O2, O3 is {self.history['adsorbates'][-1][0]}")
        prob = min(1, math.exp(-barrier / (K * TEMPERATURE_K)))
        print(f"THe prob is {prob}")
        if prob > np.random.random():
            energy = current_energy
            print(f"Action accepted")
        else:
            atoms = previous_atoms
            energy = previous_energy
            barrier = 10E-8
            # adsorbates = self.history['adsorbates'][-1]
            self.env.n_O2, self.env.n_O3 = self.history['adsorbates'][-1][0], self.history['adsorbates'][-1][1]
            print(f"Action denied")

        adsorbates = (self.env.n_O2, self.env.n_O3)

        # print(f"The current n O2 is {self.env.n_O2}, n O3 is {self.env.n_O3}")

        self.pd = nn.ZeroPad2d(padding = (0,0,0,self.max_observation_atoms-len(atoms.get_positions())))
        self.update_history(atoms, tmp_state_dict['action'], energy, barrier, adsorbates)
        if (self.history['timesteps'][-1] + 1) % self.save_every == 0:
            self.save_episode()
            self.plot_episode()

        return atoms
    
    def update_history(self, atoms:ase.Atoms, action:int, energy:float, barrier: float, adsorbates: Tuple[int, int]) -> None:
        # self.history['structures'] + [np.array(self.pd(torch.tensor(atoms.get_positions())).flatten())]
        self.history['structures'].append(np.array(self.pd(torch.tensor(atoms.get_positions())).flatten()))
        self.history['energies'].append(energy)
        self.history['actions'].append(action)
        self.history['timesteps'].append(self.history['timesteps'][-1] + 1)
        self.history['real_time'].append(self.history['real_time'][-1] + KCAL_2_EV * barrier / (R * TEMPERATURE_K))
        self.history['barriers'].append(barrier)
        self.history['adsorbates'].append(adsorbates)

    def save_episode(self) -> None:
        save_path = os.path.join(self.history_dir, "kmc_info.npz")
        np.savez_compressed(
            save_path,
            
            initial_energy=self.initial_energy,
            energies=self.history['energies'],
            actions=self.history['actions'],
            structures=self.history['structures'],
            timesteps=self.history['timesteps'],
            real_time=self.history['real_time'],
            barriers=self.history['barriers'] 
        )
        return

    def plot_episode(self) -> None:
        save_path = os.path.join(self.plot_dir, "kmc_info.png")

        energies = np.array(self.history['energies'])
        actions = np.array(self.history['actions'])

        plt.figure(figsize=(30, 30))
        plt.xlabel('steps')
        plt.ylabel('Energies')
        plt.plot(energies, color='blue')

        for action_index in range(len(ACTION_SPACES)):
            action_time = np.where(actions == action_index)[0]
            plt.plot(action_time, energies[action_time], 'o',
                     label=ACTION_SPACES[action_index])
            
        plt.legend(loc='upper left')
        plt.savefig(save_path, bbox_inches='tight')
        return plt.close('all')


    def _generate_initial_slab(self, re_read: bool = False) -> ase.Atoms:
        if os.path.exists('./input.xyz') and re_read:
            atoms = read('input.xyz')
        else:
            surfaces = [(1, 0, 0),(1, 1, 0), (1, 1, 1)]
            esurf = [1.0, 1.0, 1.0]   # Surface energies.
            lc = 3.89
            size = 147 # Number of atoms
            atoms = wulff_construction('Pd', surfaces, esurf,
                                    size, 'fcc',
                                    rounding='closest', latticeconstant=lc)

            uc = np.array([[30.0, 0, 0],
                        [0, 30.0, 0],
                        [0, 0, 30.0]])

            atoms.set_cell(uc)
        return atoms
    
    
    def iteration(self, atoms:ase.Atoms, log_dir:str, max_episodes:int, d:int,
                  map_location:Optional[str] = None, load_pkl:Optional[bool] = None) -> None:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        if load_pkl:
            atoms = torch.load(log_dir + 'model.pkl', map_location)['atoms']
            self.history = torch.load(log_dir + 'model.pkl', map_location)['state']
            self.env.n_O2, self.env.n_O3 = 2000, 0

        for i in range(d):
            with tqdm(total=int(max_episodes / d), desc='Iteration %d' % i) as pbar:
                for i_episode in range(int(max_episodes / d)):
                    atoms = self.run(atoms, max_episodes/d * i + i_episode+1)

                    checkpoint = {
                        'state' : self.history,
                        'atoms' : atoms,
                    }

                    if (max_episodes/d* i + i_episode + 1) % 100 == 0:
                        torch.save(checkpoint, log_dir + 'model.pkl')
                    
                    pbar.update(1)
                    pbar.set_postfix({'episode': '%d' % (max_episodes/d * i + i_episode+1),
                                    'energy': '%.3f' % (self.history['energies'][-1] - self.initial_energy),
                                    'action': '%d' % self.history['actions'][-1]})
    
if __name__ == '__main__':
    model = RunKMC(calculate_method='MACE', model_path = 'PdO.model', save_dir='save_dir', save_every=100)
    atoms = model.initial_slab
    print(model.env.E_O2)
    log_dir = './save_dir/save_model/'
    model.iteration(atoms, log_dir=log_dir, max_episodes=int(10E5), d=int(10E3), load_pkl = False)