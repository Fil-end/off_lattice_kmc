from typing import List,Tuple
from dataclasses import dataclass
import torch
import numpy as np

import ase
from ase import units
from ase.optimize import QuasiNewton, LBFGS, LBFGSLineSearch
from ase.units import Hartree
from ase.io.lasp_PdO import write_arc, read_arc
from ase.calculators.lasp_bulk import LASP
from ase.calculators.emt import EMT
from ase.md.langevin import Langevin

@dataclass
class Calculator:
    model_path: str = None
    calculate_method: str = None
    temperature_K: float = 473.15

    def to_calc(self, atoms:ase.Atoms, calc_type:str = 'opt'):
        if self.calculate_method in ["LASP", "Lasp", "lasp"]:
            if calc_type == "opt":
                atoms, energy, force = self.lasp_calc(atoms)
                return atoms, energy, force
            elif calc_type in ["single-point", "single"]:
                energy = self.lasp_single_calc(atoms)
                return energy
            elif calc_type in ["ssw", "SSW"]:
                energy = self.lasp_ssw_calc(atoms)
                return energy
            elif calc_type in ['MD', 'md']:
                atoms = self.lasp_md_calc(atoms)
                return atoms
            elif calc_type in ['TS', 'ts']:
                barrier = self.lasp_ts_calc(atoms)
                return barrier
            else:
                raise ValueError("No such calc type currently!!!")
            
        elif self.calculate_method in ["MACE", "Mace", "mace"]:
            if calc_type == "opt":
                atoms, energy, force = self.mace_calc(atoms)
                return atoms, energy, force
            elif calc_type in ["single-point", "single"]:
                energy = self.mace_single_calc(atoms)
                return energy
            elif calc_type in ['MD', 'md']:
                atoms = self.mace_md_calc(atoms)
                return atoms
            else:
                raise ValueError("No such calc type currently!!!")
            
        elif self.calculate_method in ["NequIP", "Nequip", "nequip"]:
            if calc_type == "opt":
                atoms, energy, force = self.nequip_calc(atoms)
                return atoms, energy, force
            elif calc_type in ["single-point", "single"]:
                energy = self.nequip_single_calc(atoms)
                return energy
            else:
                raise ValueError("No such calc type currently!!!")
        
        else:
            raise ValueError("No such calculator currently!!!")

    
    '''-------------MACE_calc--------------------'''
    def mace_calc(self, atoms:ase.Atoms) -> Tuple:
        from mace.calculators import MACECalculator
        from ase.calculators.mace_lj import MaceLjCalculator

        if self.model_path is None:
            self.model_path = 'my_mace.model'
        calculator = MACECalculator(model_paths=self.model_path, device='cuda')
        atoms.set_calculator(calculator)

        dyn = LBFGS(atoms, trajectory='lbfgs.traj')
        dyn.run(steps = 200, fmax = 0.1)

        return atoms, atoms.get_potential_energy(), atoms.get_forces()

    def mace_single_calc(self, atoms:ase.Atoms) -> int:
        from mace.calculators import MACECalculator
        from ase.calculators.mace_lj import MaceLjCalculator
        
        if self.model_path is None:
            self.model_path = 'my_mace.model'
        calculator = MACECalculator(model_paths=self.model_path, device='cuda')
        atoms.set_calculator(calculator)

        return atoms.get_potential_energy()
    
    def mace_md_calc(self, atoms:ase.Atoms) -> ase.Atoms:
        from mace.calculators import MACECalculator
        from ase.calculators.mace_lj import MaceLjCalculator

        steps = 100
        if self.model_path is None:
            self.model_path = 'my_mace.model'
        calculator = MACECalculator(model_paths=self.model_path, device='cuda')
        atoms.set_calculator(calculator)
        dyn = Langevin(atoms, 5 * units.fs, self.temperature_K * units.kB, 0.002, trajectory='md.traj',
                           logfile='MD.log')
        dyn.run(steps)
        return atoms
    
    '''-------------------LASP_calc---------------------------'''    
    def lasp_calc(self, atoms):
        write_arc([atoms])
        atoms.calc = LASP(task='opt', pot=self.model_path, potential='NN D3')
        energy = atoms.get_potential_energy()
        force = atoms.get_forces()
        atoms = read_arc('allstr.arc', index = -1)
        return atoms, energy, force
    
    def lasp_single_calc(self, atoms):
        write_arc([atoms])
        atoms.calc = LASP(task='single-energy', pot=self.model_path, potential='NN D3')
        energy = atoms.get_potential_energy()
        return energy
    
    def lasp_ssw_calc(self, atoms):
        write_arc([atoms])
        atoms.calc = LASP(task='ssw', pot=self.model_path, potential='NN D3')
        energy = atoms.get_potential_energy()
        atoms = read_arc('all.arc', index = -1)
        return atoms
    
    def lasp_md_calc(self, atoms):
        steps = 100
        atoms.calc = EMT()
        dyn = Langevin(atoms, 5 * units.fs, self.temperature_K * units.kB, 0.002, trajectory='md.traj',
                           logfile='MD.log')
        dyn.run(steps)
        return atoms
    
    def lasp_ts_calc(self, atoms:List[ase.Atoms]):
        write_arc(atoms[0])
        write_arc(atoms)
        atoms[0].calc = LASP(task='TS', pot=self.model_path, potential='NN D3')
        if atoms[0].get_potential_energy() == 0:  #没有搜索到过渡态
            barrier = 0
        else:
            barrier, _ = atoms[0].get_potential_energy()

        return barrier

    '''----------------------Nequip_calc--------------------------'''
    def nequip_calc(self, atoms):
        import nequip
        from nequip.ase import NequIPCalculator

        calc = NequIPCalculator.from_deployed_model(model_path=self.model_path,
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
            species_to_type_name = {'O': 'O', self.metal_ele: self.metal_ele},
                )
        
        atoms.calc = calc
        dyn = LBFGS(atoms, trajectory='lbfgs.traj')
        dyn.run(steps = 200, fmax = 0.05)

        return atoms, atoms.get_potential_energy(), atoms.get_forces()
    
    def nequip_single_calc(self, atoms):

        import nequip
        from nequip.ase import NequIPCalculator

        calc = NequIPCalculator.from_deployed_model(model_path=self.model_path,
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
            species_to_type_name = {'O': 'O', self.metal_ele: self.metal_ele},
                )
        
        atoms.calc = calc

        return atoms.get_potential_energy()