from operator import ne
from tools.periodic_table import ELERADII
from tools.calc import Calculator
from tools.cluster_actions import ClusterActions

import os
import numpy as np
import math
from math import cos, sin
from scipy.spatial import Delaunay
from typing import List, Optional

import ase
from ase import Atom
from ase.build import molecule
from ase.cluster.wulff import wulff_construction
from ase.constraints import FixAtoms
from ase.geometry.analysis import Analysis
from ase.io import read
from ase.io.lasp_PdO import write_arc, read_arc


# 设定动作空间
ACTION_SPACES = ['ADS', 'Translation', 'R_Rotation', 'L_Rotation', 'MD', 'Diffusion', 'Drill', 'Dissociation', 'Desportion']
r_O = ELERADII[7]
r_Si = ELERADII[14]
r_Pd = ELERADII[45]

d_O_Pd = r_O + r_Pd
d_O_O = 2 * r_O
d_Pd_Pd = 2 * r_Pd
d_Si_Pd = r_Si + r_Pd


class OffLatticeKMC():
    '''
        This class is use to construct a off-lattice KMC environment

        Args:
            calculate_method(str): The calculate method define which calculator we would use in
            the following project. Currrently, we can only use calculator in ['MACE', 'LASP']
            cutoff()
    '''
    def __init__(self, 
                 initial_slab: Optional[ase.Atoms] = None, 
                 calculator: Optional[Calculator] = None, 
                 cutoff: float = 4.0,
                 num_adsorbates:Optional[List] = None,
                 delta_s:int = -0.371,
                 in_zeolite: bool = False,
                 cluster_metal = "Pd"):
        # initialize the initial cluster
        if not initial_slab:
            self.initial_slab = self._generate_initial_slab(in_zeolite)
        else:
            self.initial_slab = initial_slab

        self.cutoff = cutoff
        self.cluster_metal = cluster_metal
        self.delta_s = delta_s
        self.in_zeolite = in_zeolite

        # initialize the calculator
        if calculator:
            self.calculator = calculator
        else:
            self.calculator = Calculator(calculate_method='MACE')

        # initialize ClusterAction class
        self.cluster_actions = ClusterActions(metal_ele = self.cluster_metal)
        
        # pre_processing the Zeolite system
        if self.in_zeolite:
            self.system = self.initial_slab.copy()
            self.zeolite = self.system.copy()
            del self.zeolite[[a.index for a in self.zeolite if a.symbol == self.cluster_metal]]
            self.cluster = self.system.copy()
            del self.cluster[[a.index for a in self.cluster if a.symbol != self.cluster_metal]]
            self.system = self.zeolite + self.cluster

            self.initial_slab = self.cluster.copy()

        else:
            self.system = self.cluster_actions.rectify_atoms_positions(self.initial_slab)

        # get all surfaces
        self.total_surfaces = self._mock_cluster().get_surfaces()

        self.free_list = self._get_free_atoms_list(self.system)
        self.fix_list = [atom_idx for atom_idx in range(len(self.system)) if atom_idx not in self.free_list]

        # calculate the energy of single molecule
        self.n_O2 = 2000
        self.n_O3 = 0

        self.ads_list = []
        for _ in range(self.n_O2):
            self.ads_list.append(2)

        self.E_O2 = self.add_mole("O2")
        self.E_O3 = self.add_mole("O3")

    def generate_neigh_atoms(self, atoms:ase.Atoms, target_site:np.ndarray) -> List[int]:
        neigh_atom_index_list = []
        for atom_idx in range(len(atoms)):
            atom = atoms[atom_idx]
            d = self.distance(target_site[0], target_site[1], target_site[2],
                              atom.position[0], atom.position[1], atom.position[2])
            
            if d < self.cutoff:
                neigh_atom_index_list.append(atom_idx)

        return neigh_atom_index_list

    def generate_action_list(self, atoms:ase.Atoms, target_site:np.ndarray, vector: List) -> List:
        '''
            This function is to identify which action can occur near the selected_site
            Args:
                atoms
            Return:
                current_action_list(List): Actions could do near the current site
        '''
        action_list = []
        neigh_atom_index_list = self.generate_neigh_atoms(atoms, target_site)
        if self._can_adsorb(atoms, target_site, neigh_atom_index_list):
            action_list.append(0)
        if self._can_diffuse(atoms, neigh_atom_index_list):
            action_list.extend([5, 6])
        if self._can_desorb(atoms, neigh_atom_index_list):
            action_list.extend([7,8])    
        
        return action_list

    def step(self, atoms:ase.Atoms, previous_energy: float, episode: int, all_action:bool =False):
        '''
            This part is the main part of the off-lattice kmc project, and it will interact
            with the main.py
        '''
        self.facet_selection = self.total_surfaces[np.random.randint(len(self.total_surfaces))]
        if self.in_zeolite:
            self.center_point = self.cluster_actions.get_center_point(self._get_cluster(atoms, with_zeolite=self.in_zeolite))
        else:
            self.center_point = self.cluster_actions.get_center_point(atoms)

        atoms = self.cluster_actions.cluster_rotation(atoms, self.facet_selection, self.center_point)
        # total_layer_O_list, total_sub_O_list = self.get_O_info(atoms)

        surfList = self.get_surf_atoms(atoms)
        surf_metal_list = self.get_surf_metal_atoms(atoms, surfList)

        print(f"The initial_len(surf_metal_list is) {len(surf_metal_list)}")
        # if len(surf_metal_list) > 3:
        surf_sites, surf_plane_vector = self.get_surf_sites(atoms)
        # else:
        #     atoms = self.cluster_actions.recover_rotation(atoms, self.facet_selection, self.center_point)
        #     addable_facet_list = []
        #     prior_ads_list = []
        #    for facet in self.total_surfaces:
        #         atoms = self.cluster_actions.cluster_rotation(atoms, facet, self.center_point)
        #         list = self.get_surf_atoms(atoms)
        #         surf_metal_list_tmp = self.get_surf_metal_atoms(atoms, list)
        #         layer_list = self.get_layer_atoms(atoms)

        #         if len(surf_metal_list_tmp) > 3:
        #             for i in layer_list + list:
        #                 if i not in total_layer_O_list and i not in total_sub_O_list:
        #                     prior_ads_list.append(facet)

        #             addable_facet_list.append(facet)
        #         atoms = self.cluster_actions.recover_rotation(atoms, facet, self.center_point)
        #     if prior_ads_list:
        #         self.facet_selection = prior_ads_list[np.random.randint(len(prior_ads_list))]
        #     else:
        #         self.facet_selection = addable_facet_list[np.random.randint(len(addable_facet_list))]
        #     atoms = self.cluster_actions.cluster_rotation(atoms, self.facet_selection, self.center_point)

        #     surf_sites, surf_plane_vector = self.get_surf_sites(atoms)


        site_idx = np.random.randint(len(surf_sites))
        selected_site = surf_sites[site_idx]
        selected_vector = surf_plane_vector[site_idx]

        action_list = self.generate_action_list(atoms, selected_site, selected_vector)
        if not action_list or episode % 100 == 0:
            action_list = [1,2,3]

        print(f"The current action_list is {action_list}")

        tmp_state_dict = {}
        results = ['energies', 'actions', 'atoms', 'barriers', 'probabilites', 'adsorbates', 'zeolite']
        for item in results:
            tmp_state_dict[item] = []

        tmp_n_O2, tmp_n_O3 = self.n_O2, self.n_O3
        
        # Iterate all possible 
        # center_point = self.cluster_actions.get_center_point(atoms)

        for action_idx in action_list:
            self.n_O2, self.n_O3 = tmp_n_O2, tmp_n_O3
            new_state = atoms.copy()
            write_arc([new_state], name = "chk_pt.arc")
            new_state = self.choose_actions(new_state, action_idx, selected_site, selected_vector)   # here new_state is cluster
            new_state = self.cluster_actions.rectify_atoms_positions(new_state, self.center_point)

            write_arc([new_state], name = "chk_pt_1.arc")
            self.to_constraint(new_state, with_zeolite=self.in_zeolite)
            new_state, energy, _ = self.calculator.to_calc(new_state)

            O_list = []
            for atom in new_state:
                if atom.symbol == 'O':
                    O_list.append(atom.index)

            current_energy = energy + self.n_O2 * self.E_O2 + self.n_O3 * self.E_O3

            if action_idx == 0:
                current_energy = current_energy - self.delta_s

            if action_idx == 8:
                current_energy = current_energy + self.delta_s

            barrier = self.transition_state_search(previous_energy, current_energy, action_idx)
            if barrier == 0:
                barrier += 10E-8
            
            if current_energy - previous_energy > -100:
                tmp_state_dict['atoms'].append(new_state)
                tmp_state_dict['energies'].append(current_energy)
                tmp_state_dict['actions'].append(action_idx)
                tmp_state_dict['barriers'].append(barrier)
                tmp_state_dict['probabilites'].append(1 / barrier)
                tmp_state_dict['adsorbates'].append((self.n_O2, self.n_O3))

        prob_list = self.get_prob_list(tmp_state_dict['probabilites'])
        prob = np.random.rand()

        if bool(sum(prob_list)):    # 如果prob_list = [0.0, 0.0, 0.0, 0.0], 则直接停止操作，并且reward -= 5
            for i in range(len(prob_list) - 1):
                if prob_list[i - 1] < prob and prob < prob_list[i]:
                    selected_idx = i
                    break
        
        if tmp_state_dict['atoms']:
            state_dict = {}
            atoms = tmp_state_dict['atoms'][selected_idx]
            state_dict['structure'] = tmp_state_dict['atoms'][selected_idx].get_positions()
            state_dict['energy'] = tmp_state_dict['energies'][selected_idx]
            state_dict['action'] = tmp_state_dict['actions'][selected_idx]
            state_dict['probability'] = tmp_state_dict['probabilites'][selected_idx]
            state_dict['barrier'] = tmp_state_dict['barriers'][selected_idx]
            
            self.n_O2, self.n_O3 = tmp_state_dict['adsorbates'][selected_idx]
        else:
            state_dict = {}
            state_dict['structure'] = atoms.get_positions()
            state_dict['energy'] = previous_energy
            state_dict['action'] = 4
            state_dict['probability'] = 0.0
            state_dict['barrier'] = 0.0

        print(f"The current action is {state_dict['action']}, current n O2 = {self.n_O2}, current n O3 = {self.n_O3}")

        return atoms, state_dict

    def choose_actions(self, atoms:ase.Atoms, action_idx:str, selected_site:np.ndarray, selected_vector:List) -> ase.Atoms:
        if action_idx in [1,2,3,8]:
            atoms = self._get_cluster(atoms, with_zeolite=self.in_zeolite)

        self.to_constraint(atoms)
        if action_idx == 0:
            atoms = self._adsorb(atoms, selected_site, selected_vector)

        elif action_idx == 1:
            self._translate(atoms)

        elif action_idx == 2:
            self._rotate(atoms, 3)

        elif action_idx == 3:
            self._rotate(atoms, -3)

        elif action_idx == 4:
            self._md(atoms)

        #------------The above actions are muti-actions and the following actions contain single-atom actions--------------------------------
        elif action_idx == 5:  # 表面上氧原子的扩散，单原子行为
            atoms = self._diffuse(atoms, selected_site, selected_vector)

        elif action_idx == 6:  # 表面晶胞的扩大以及氧原子的钻洞，多原子行为+单原子行为
            atoms = self._drill(atoms, selected_site)

        elif action_idx == 7:  # 氧气解离
            atoms = self._dissociate(atoms, selected_site)

        elif action_idx == 8:
            atoms= self._desorb(atoms, selected_site)
            
        else:
            print('No such action')

        atoms = self.cluster_actions.recover_rotation(atoms, self.facet_selection, self.center_point)

        if action_idx in [1,2,3,8]:
            atoms = self._get_system(atoms, with_zeolite=self.in_zeolite)

        return atoms

    def _generate_initial_slab(self, re_read = False) -> ase.atoms:
        if os.path.exists('./initial_input.arc') and re_read:
            atoms = read_arc('initial_input.arc')[0]
        else:
            atoms = self._mock_cluster()
        return atoms    
    
    def _mock_cluster(self) -> ase.Atoms:
        if os.path.exists('./mock.xyz'):
            atoms = read('mock.xyz')
        else:
            surfaces = [(1, 0, 0),(1, 1, 0), (1, 1, 1)]
            esurf = [1.0, 1.0, 1.0]   # Surface energies.
            lc = 3.89
            size = 147 # Number of atoms
            atoms = wulff_construction(self.cluster_metal, surfaces, esurf,
                                    size, 'fcc',
                                    rounding='closest', latticeconstant=lc)

            uc = np.array([[30.0, 0, 0],
                            [0, 30.0, 0],
                            [0, 0, 30.0]])

            atoms.set_cell(uc)
        return atoms
    
    def get_constraint(self, atoms:ase.Atoms, with_zeolite:bool = False) -> FixAtoms:
        if with_zeolite:
            constrain_list = self.fix_list
            constraint = FixAtoms(mask=[a.index in constrain_list for a in atoms]) 
        else:
            surfList = []
            for facet in self.total_surfaces:
                atoms= self.cluster_actions.cluster_rotation(atoms, facet, self.center_point)
                surf_list = self.get_surf_atoms(atoms)
                for i in surf_list:
                    surfList.append(i)
                atoms = self.cluster_actions.recover_rotation(atoms, facet, self.center_point)
            constrain_list = [i for n, i in enumerate(surfList) if i not in surfList[:n]]
            # print(f"The current constrain list is {constrain_list}")
            constraint = FixAtoms(mask=[a.symbol != 'O' and a.index in constrain_list for a in atoms])
        return constraint
    
    def to_constraint(self, atoms:ase.Atoms, with_zeolite:bool = False) -> None: # depending on such type of atoms
        # print(f"The current atoms is {atoms}")
        constraint = self.get_constraint(atoms, with_zeolite)
        atoms.set_constraint(constraint)
    
    '''---------The following code will be the main actions this project used---------'''
    def get_ads_d(self, ads_site):
        if ads_site[3] == 1:
            d = 1.5
        elif ads_site[3] == 2:
            d = 1.3
        else:
            d = 1.0
        return d
    
    def _adsorb(self, atoms: ase.Atoms, selected_site:np.ndarray, ads_vector = None) -> ase.Atoms:
        new_state = atoms.copy() 
        ads_site = selected_site
        choosed_adsorbate = np.random.randint(len(self.ads_list))
        ads = self.ads_list[choosed_adsorbate]
            
        del self.ads_list[choosed_adsorbate]
        if ads:
            d = 1.5
            if ads == 2:
                self.n_O2 -= 1
                d = self.get_ads_d(ads_site)
                O1 = Atom('O', (ads_site[0], ads_site[1], ads_site[2] + d))
                O2 = Atom('O', (ads_site[0], ads_site[1], ads_site[2] + (d + 1.21)))
                new_state = new_state + O1
                new_state = new_state + O2

            elif ads == 3:
                self.n_O3 -= 1
                O1 = Atom('O', (ads_site[0], ads_site[1], ads_site[2] + d))
                O2 = Atom('O', (ads_site[0], ads_site[1] + 1.09, ads_site[2] + d + 0.67))
                O3 = Atom('O', (ads_site[0], ads_site[1] - 1.09, ads_site[2] + d + 0.67))
                new_state = new_state + O1
                new_state = new_state + O2
                new_state = new_state + O3

        return new_state
    
    def _translate(self, atoms:ase.Atoms) -> None:
        lamada_d, lamada_s, lamada_layer = 0.2, 0.4, 0.6

        layer_atom, surf_atom, sub_atom, deep_atom = self.get_atom_info(atoms)
        muti_movement = np.array([np.random.normal(0.25,0.25), np.random.normal(0.25,0.25), np.random.normal(0.25,0.25)])
        initial_positions = atoms.positions

        for atom in initial_positions:
            if atom in deep_atom:
                atom += lamada_d * muti_movement
            if atom in sub_atom:
                atom += lamada_s * muti_movement
            if atom in surf_atom:
                atom += lamada_layer * muti_movement
            if atom in layer_atom:
                atom += lamada_layer * muti_movement
        atoms.positions = initial_positions

    def _get_rotate_matrix(self, zeta:float) -> np.ndarray[List]:
        matrix = [[cos(zeta), -sin(zeta), 0],
                      [sin(zeta), cos(zeta), 0],
                      [0, 0, 1]]
        return np.array(matrix)
    
    def _rotate(self, atoms:ase.Atoms, zeta:float) -> None:
        initial_state = atoms.copy()

        zeta = math.pi * zeta / 180
        surf_matrix = self._get_rotate_matrix(zeta * 3)
        sub_matrix = self._get_rotate_matrix(zeta * 2)
        deep_matrix = self._get_rotate_matrix(zeta)

        rotation_surf_list = []
        rotation_deep_list = []
        rotation_sub_list = []

        surf_list = self.get_surf_atoms(atoms)
        layer_list = self.get_layer_atoms(atoms)
        sub_list = self.get_sub_atoms(atoms)
        deep_list = self.get_deep_atoms(atoms)

        for i in surf_list:
            rotation_surf_list.append(i)
        for j in layer_list:
            rotation_surf_list.append(j)

        for k in sub_list:
            rotation_sub_list.append(k)

        for m in deep_list:
            rotation_deep_list.append(m)

        rotation_surf_list = [i for n, i in enumerate(rotation_surf_list) if i not in rotation_surf_list[:n]]

        central_point = self.mid_point(atoms, surf_list)

        for atom in initial_state:

            if atom.index in rotation_surf_list:
                atom.position += np.array(
                        (np.dot(surf_matrix, (np.array(atom.position.tolist()) - central_point).T).T + central_point).tolist()) - atom.position
            elif atom.index in rotation_sub_list:
                atom.position += np.array(
                        (np.dot(sub_matrix, (np.array(atom.position.tolist()) - central_point).T).T + central_point).tolist()) - atom.position
            elif atom.index in rotation_deep_list:
                atom.position += np.array(
                        (np.dot(deep_matrix, (np.array(atom.position.tolist()) - central_point).T).T + central_point).tolist()) - atom.position
                
        atoms.positions = initial_state.get_positions()

    def _md(self, atoms:ase.Atoms):
        if self.in_zeolite:
            atoms = self._get_system(atoms, with_zeolite=self.in_zeolite)

        if self.calculator.calculate_method in ["MACE", "Mace", "mace"]:
            atoms = self.calculator.to_calc(atoms, 'MD')
        elif self.calculator.calculate_method in ["LASP", "Lasp", "lasp"]:
            atoms = self.calculator.to_calc(atoms, 'ssw')

        if self.in_zeolite:
            atoms = self._get_cluster(atoms, with_zeolite=self.in_zeolite)

    def _diffuse(self, slab:ase.Atoms, selected_site: np.ndarray, diffuse_vector:List) -> ase.Atoms:
        total_layer_O, _ = self.get_O_info(slab)
        if total_layer_O:
            to_diffuse_O_list = []

            neigh_atom_index_list = self.generate_neigh_atoms(slab, selected_site)
            single_layer_O = self.layer_O_atom_list(slab)

            for atom_idx in neigh_atom_index_list:
                if slab[atom_idx].symbol == 'O' and atom_idx in single_layer_O:
                    to_diffuse_O_list.append(atom_idx)

            selected_O_index = to_diffuse_O_list[np.random.randint(len(to_diffuse_O_list))]
            diffuse_site = selected_site
            
            d = 1.5
            for atom in slab:
                if atom.index == selected_O_index:
                    d = self.get_ads_d(diffuse_site)
                    atom.position = np.array([diffuse_site[0] + diffuse_vector[0] * d, 
                                              diffuse_site[1] + diffuse_vector[1] * d, 
                                              diffuse_site[2] + diffuse_vector[2] * d])
            
        return slab
    
    def _drill(self, slab:ase.Atoms, selected_site:np.ndarray) -> ase.Atoms:
        total_layer_O, _ = self.get_O_info(slab)

        if total_layer_O:
            selected_drill_O_list = []
            single_layer_O = self.layer_O_atom_list(slab)

            neigh_atom_index_list = self.generate_neigh_atoms(slab, selected_site)

            for atom_idx in neigh_atom_index_list:
                if slab[atom_idx].symbol == 'O' and atom_idx in single_layer_O:
                    selected_drill_O_list.append(atom_idx)

            if selected_drill_O_list:
                selected_O = selected_drill_O_list[np.random.randint(len(selected_drill_O_list))]

                if selected_O in single_layer_O:
                    slab, action_done = self.to_drill_surf(slab, selected_O)

        return slab

    def lifted_distance(self, drill_site: np.ndarray, pos: np.ndarray) -> float:

        r = self.distance(drill_site[0], drill_site[1], drill_site[2] +1.3,
                                    pos[0], pos[1], pos[2])
        
        lifted_d = math.exp(- r * r / (2 * 2.5 ** 2))

        return min(lifted_d, 0.5)
    
    def to_drill_surf(self, slab: ase.Atoms, selected_drill_atom: int) -> ase.Atoms:
        action_done = True

        layer_O = []
        to_distance = []
        drillable_sites = []
        layer_List = self.get_layer_atoms(slab)

        sub_sites, sub_plane_vector = self.get_sub_sites(slab)

        for i in slab:
            if i.index in layer_List and i.symbol == 'O':
                layer_O.append(i.index)
        
        for ads_sites in sub_sites:
            to_other_O_distance = []
            if layer_O:
                for i in layer_O:
                    distance = self.distance(ads_sites[0], ads_sites[1], ads_sites[2] + 1.3, slab.get_positions()[i][0],
                                           slab.get_positions()[i][1], slab.get_positions()[i][2])
                    to_other_O_distance.append(distance)
                if min(to_other_O_distance) > 2 * d_O_O:
                    ads_sites[4] = 1
                else:
                    ads_sites[4] = 0
            else:
                ads_sites[4] = 1
            if ads_sites[4]:
                drillable_sites.append(ads_sites)

        position = slab.get_positions()[selected_drill_atom]
        for drill_site in drillable_sites:
                to_distance.append(
                            self.distance(position[0], position[1], position[2], drill_site[0], drill_site[1],
                                        drill_site[2]))

        if to_distance:
            drill_site = sub_sites[to_distance.index(min(to_distance))]
            
            for atom in slab:
                if atom.index == selected_drill_atom:
                    atom.position = np.array([drill_site[0], drill_site[1], drill_site[2] +1.0])

            lifted_atoms_list = []
            current_surfList = self.get_surf_atoms(slab)
            current_layerList = self.get_layer_atoms(slab)

            for layer_atom in current_layerList:
                lifted_atoms_list.append(layer_atom)

            for surf_atom in current_surfList:
                lifted_atoms_list.append(surf_atom)

            for lifted_atom in lifted_atoms_list:
                slab.positions[lifted_atom][2] += self.lifted_distance(drill_site, slab.get_positions()[lifted_atom])
        else:
            action_done = False
        return slab, action_done
    
    def _dissociate(self, slab:ase.Atoms, selected_site:np.ndarray) -> ase.Atoms:
        action_done = True

        dissociate_O2_list = []        
        all_dissociate_O2_list = self.get_dissociate_O2_list(slab)
        neigh_atom_index_list = self.generate_neigh_atoms(slab, selected_site)

        for mole in all_dissociate_O2_list:
            if mole[0] in neigh_atom_index_list and mole[1] in neigh_atom_index_list:
                dissociate_O2_list.append(mole)

        if dissociate_O2_list:
            OO = dissociate_O2_list[np.random.randint(len(dissociate_O2_list))]
            slab = self.oxy_rotation(slab, OO)
            slab, action_done = self.to_dissociate(slab, OO)
        else:
            action_done = False

        return slab
    
    def ball_func(self,pos1, pos2):	# zeta < 36, fi < 3
        d = self.distance(pos1[0],pos1[1],pos1[2],pos2[0],pos2[1],pos2[2])
        '''如果pos1[2] > pos2[2],atom_1旋转下来'''
        pos2_position = pos2
        pos_slr = pos1 - pos2

        pos_slr_square = math.sqrt(pos_slr[0] * pos_slr[0] + pos_slr[1] * pos_slr[1])
        pos1_position = [pos2[0] + d * pos_slr[0]/pos_slr_square, pos2[1] + d * pos_slr[1]/pos_slr_square, pos2[2]]

        return pos1_position, pos2_position
    
    def oxy_rotation(self, slab:ase.Atoms, OO:ase.Atoms):
        if slab.positions[OO[0][0]][2] > slab.positions[OO[0][1]][2]:
            a,b = self.ball_func(slab.get_positions()[OO[0][0]], slab.get_positions()[OO[0][1]])
        else:
            a,b = self.ball_func(slab.get_positions()[OO[0][1]], slab.get_positions()[OO[0][0]])
        slab.positions[OO[0][0]] = a
        slab.positions[OO[0][1]] = b
        return slab
    
    def to_dissociate(self, slab, atoms):
        action_done = True

        expanding_index = 2.0
        central_point = np.array([(slab.get_positions()[atoms[0][0]][0] + slab.get_positions()[atoms[0][1]][0])/2, 
                                  (slab.get_positions()[atoms[0][0]][1] + slab.get_positions()[atoms[0][1]][1])/2, 
                                  (slab.get_positions()[atoms[0][0]][2] + slab.get_positions()[atoms[0][1]][2])/2])
        slab.positions[atoms[0][0]] += np.array([expanding_index*(slab.get_positions()[atoms[0][0]][0]-central_point[0]), 
                                                 expanding_index*(slab.get_positions()[atoms[0][0]][1]-central_point[1]), 
                                                 expanding_index*(slab.get_positions()[atoms[0][0]][2]-central_point[2])])
        slab.positions[atoms[0][1]] += np.array([expanding_index*(slab.get_positions()[atoms[0][1]][0]-central_point[0]), 
                                                 expanding_index*(slab.get_positions()[atoms[0][1]][1]-central_point[1]), 
                                                 expanding_index*(slab.get_positions()[atoms[0][1]][2]-central_point[2])])
        
        addable_sites = []
        layer_O = []
        layerlist = self.get_layer_atoms(slab)

        surfList = self.get_surf_atoms(slab)
        surf_metal_list = self.get_surf_metal_atoms(slab, surfList)
        print(f"The num of surf metal is {surf_metal_list}")
        if len(surf_metal_list) > 3:
            surf_sites, surf_plane_vector = self.get_surf_sites(slab)
        else:
            neigh_facets = self.neighbour_facet(slab, self.facet_selection)
            # now the facet has been recovered
            to_dissociate_facet_list = []
            for facet in neigh_facets:
                slab = self.cluster_actions.cluster_rotation(slab, facet, self.center_point)
                neigh_surf_list = self.get_surf_atoms(slab)
                neigh_surf_metal_list = self.get_surf_metal_atoms(slab, neigh_surf_list)
                if len(neigh_surf_metal_list) > 3:
                   to_dissociate_facet_list.append(facet)
                slab = self.cluster_actions.recover_rotation(slab, facet, self.center_point)
            
            if to_dissociate_facet_list:
                self.facet_selection = to_dissociate_facet_list[np.random.randint(len(to_dissociate_facet_list))]

            slab = self.cluster_actions.cluster_rotation(slab, self.facet_selection, self.center_point)
            surf_sites, surf_plane_vector = self.get_surf_sites(slab)

        for ads_site in surf_sites:
            for atom_index in layerlist:
                if slab[atom_index].symbol == 'O':
                    layer_O.append(atom_index)
            to_other_O_distance = []
            if layer_O:
                for i in layer_O:
                    to_distance = self.distance(ads_site[0], ads_site[1], ads_site[2] + 1.5, slab.get_positions()[i][0],
                                           slab.get_positions()[i][1], slab.get_positions()[i][2])
                    to_other_O_distance.append(to_distance)
                if min(to_other_O_distance) > 1.5 * d_O_O:
                    ads_site[4] = 1
            else:
                ads_site[4] = 1
            if ads_site[4]:
                addable_sites.append(ads_site)

        O1_distance = []
        for add_1_site in addable_sites:
            distance_1 = self.distance(add_1_site[0], add_1_site[1], add_1_site[2] + 1.3, slab.get_positions()[atoms[0][0]][0],
                                           slab.get_positions()[atoms[0][0]][1], slab.get_positions()[atoms[0][0]][2])
            O1_distance.append(distance_1)

        if O1_distance:
            O1_site = addable_sites[O1_distance.index(min(O1_distance))]
            
            ad_2_sites = []
            for add_site in addable_sites:
                d = self.distance(add_site[0], add_site[1], add_site[2] + 1.3, O1_site[0], O1_site[1], O1_site[2])
                if d > 2.0 * d_O_O:
                    ad_2_sites.append(add_site)

            O2_distance = []
            for add_2_site in ad_2_sites:
                distance_2 = self.distance(add_2_site[0], add_2_site[1], add_2_site[2] + 1.3, slab.get_positions()[atoms[0][1]][0],
                                            slab.get_positions()[atoms[0][1]][1], slab.get_positions()[atoms[0][1]][2])
                O2_distance.append(distance_2)
            
            if O2_distance:
                O2_site = ad_2_sites[O2_distance.index(min(O2_distance))]
            else:
                O2_site = O1_site
            
            d_1, d_2 = 1.5, 1.5

            d_1 = self.get_ads_d(O1_site)
            d_2 = self.get_ads_d(O2_site)

            for atom in slab:
                if O1_site[0] == O2_site[0] and O1_site[1] == O2_site[1]:
                    
                    O_1_position = np.array([O1_site[0], O1_site[1], O1_site[2] + d_1])
                    O_2_position = np.array([O1_site[0], O1_site[1], O1_site[2] + d_1 + 1.21])
                    action_done = False
                else:
                    O_1_position = np.array([O1_site[0], O1_site[1], O1_site[2] + d_1])
                    O_2_position = np.array([O2_site[0], O2_site[1], O2_site[2] + d_2])

                if atom.index == atoms[0][0]:
                    atom.position = O_1_position
                elif atom.index == atoms[0][1]:
                    atom.position = O_2_position

        else:
            action_done = False
        return slab, action_done
    
    def _desorb(self, slab:ase.Atoms, selected_site:np.ndarray) -> ase.Atoms:
        action_done = True
        new_state = slab.copy()
        
        desorb_list = []     
        neigh_atom_index_list = self.generate_neigh_atoms(slab, selected_site)

        _,  all_desorb_list = self.to_desorb_adsorbate(slab)
        for mole in all_desorb_list:
            if len(mole) == 2:
                if mole[0] in neigh_atom_index_list and mole[1] in neigh_atom_index_list:
                    self.ads_list.append(2)
                    desorb_list.append(mole)
            elif len(mole) == 3:
                if mole[0] in neigh_atom_index_list and mole[1] in neigh_atom_index_list and mole[2] in neigh_atom_index_list:
                    self.ads_list.append(3)
                    desorb_list.append(mole)

        if desorb_list:
            desorb = desorb_list[np.random.randint(len(desorb_list))]
            del new_state[[i for i in range(len(new_state)) if i in desorb]]

            if len(desorb):
                if len(desorb) == 2:
                    self.n_O2 += 1

                elif len(desorb) == 3:
                    self.n_O3 += 1

            action_done = False
        action_done = False

        return new_state
    
    def get_dissociate_O2_list(self, slab: ase.Atoms) -> List:
        slab = self._get_cluster(slab, with_zeolite=self.in_zeolite)
        ana = Analysis(slab)
        OOBonds = ana.get_bonds('O','O',unique = True)
        PdOBonds = ana.get_bonds(self.cluster_metal, 'O', unique = True)

        Pd_O_list = []
        dissociate_O2_list = []

        if PdOBonds[0]:
            for i in PdOBonds[0]:
                Pd_O_list.append(i[0])
                Pd_O_list.append(i[1])

        if OOBonds[0]:
            layerList = self.get_layer_atoms(slab)
            for j in OOBonds[0]:
                if (j[0] in layerList or j[1] in layerList) and (j[0] in Pd_O_list or j[1] in Pd_O_list):
                    dissociate_O2_list.append([(j[0],j[1])])
            
        dissociate_O2_list = self._map_zeolite(dissociate_O2_list, with_zeolite=self.in_zeolite)

        return dissociate_O2_list
    
    def to_desorb_adsorbate(self, slab):
        desorb = ()
        ana = Analysis(slab)
        OOBonds = ana.get_bonds('O', 'O', unique = True)
        PdOBonds = ana.get_bonds(self.cluster_metal, 'O', unique=True)

        OOOangles = ana.get_angles('O', 'O', 'O',unique = True)

        Pd_O_list = []
        desorb_list = []
        if PdOBonds[0]:
            for i in PdOBonds[0]:
                Pd_O_list.append(i[0])
                Pd_O_list.append(i[1])
        
        if OOBonds[0]:  # 定义环境中的氧气分子
            for i in OOBonds[0]:
                if i[0] in Pd_O_list or i[1] in Pd_O_list:
                    desorb_list.append(i)

        if OOOangles[0]:
            for j in OOOangles[0]:
                if j[0] in Pd_O_list or j[1] in Pd_O_list or j[2] in Pd_O_list:
                    desorb_list.append(j)

        if desorb_list:
            desorb = desorb_list[np.random.randint(len(desorb_list))]
        return desorb, desorb_list
    
    '''------------Following part will be some layer-based small tools------------------'''
    '''------------This part will gain some layer information---------------------------'''
    def label_atoms(self, atoms:ase.Atoms, zRange:List) -> List:
        myPos = atoms.get_positions()
        return [
            i for i in range(len(atoms)) \
            if min(zRange) < myPos[i][2] < max(zRange)
        ]
    
    def modify_slab_layer_atoms(self, atoms:ase.Atoms, list:List):
        sum_z = 0
        if list:
            for i in list:
                sum_z += atoms.get_positions()[i][2]

            modified_z = sum_z / len(list)
            modified_list = self.label_atoms(atoms, [modified_z - 1.0, modified_z + 1.0])
            return modified_list
        else:
            return list
        
    def get_layer_atoms(self, atoms:ase.Atoms) -> List:
        z_list = []
        for i in range(len(atoms)):
            if atoms[i].symbol == self.cluster_metal:
                z_list.append(atoms.get_positions()[i][2])
        z_max = max(z_list)

        layerlist = self.label_atoms(atoms, [z_max - 1.0, z_max + 6.0])

        sum_z = 0
        if layerlist:
            for i in layerlist:
                sum_z += atoms.get_positions()[i][2]

            modified_z = sum_z / len(layerlist)
            modified_layer_list = self.label_atoms(atoms, [modified_z - 1.0, modified_z + 1.0])
        else:
            modified_layer_list = layerlist
        
        return modified_layer_list
    
    def get_surf_metal_atoms(self, atoms:ase.Atoms, surfList:List) -> List:
        surf_metal_list = []
        if surfList:
            for index in surfList:
                if atoms[index].symbol == self.cluster_metal:
                    surf_metal_list.append(index)

        return surf_metal_list
    
    def get_surf_atoms(self, atoms:ase.Atoms) -> List:
        z_list = []
        for i in range(len(atoms)):
            if atoms[i].symbol == self.cluster_metal:
                z_list.append(atoms.get_positions()[i][2])
        z_max = max(z_list)
        surf_z = z_max - r_Pd / 2

        surflist = self.label_atoms(atoms, [surf_z - 1.0, surf_z + 1.0])
        modified_surflist = self.modify_slab_layer_atoms(atoms, surflist)

        return modified_surflist
    
    def get_sub_atoms(self, atoms:ase.Atoms) -> List:
        z_list = []
        for i in range(len(atoms)):
            if atoms[i].symbol == self.cluster_metal:
                z_list.append(atoms.get_positions()[i][2])
        z_max = max(z_list)

        sub_z = z_max - r_Pd/2 - 2.0

        sublist = self.label_atoms(atoms, [sub_z - 1.0, sub_z + 1.0])
        modified_sublist = self.modify_slab_layer_atoms(atoms, sublist)

        return modified_sublist
    
    def get_deep_atoms(self, atoms:ase.Atoms) -> List:
        z_list = []
        for i in range(len(atoms)):
            if atoms[i].symbol == self.cluster_metal:
                z_list.append(atoms.get_positions()[i][2])
        z_max = max(z_list)
        deep_z = z_max - r_Pd/2 - 4.0

        deeplist = self.label_atoms(atoms, [deep_z - 1.0, deep_z + 1.0])
        modified_deeplist = self.modify_slab_layer_atoms(atoms, deeplist)

        return modified_deeplist
    
    def get_surf_sites(self, atoms):
        surfList = self.get_surf_atoms(atoms)

        surf = atoms.copy()
        del surf[[i for i in range(len(surf)) if (i not in surfList) or surf[i].symbol != self.cluster_metal]]
        

        surf_sites, surf_plane_vector = self.get_sites(surf)

        return surf_sites, surf_plane_vector
    
    def get_sub_sites(self, atoms):
        subList = self.get_sub_atoms(atoms)

        sub = atoms.copy()
        del sub[[i for i in range(len(sub)) if (i not in subList) or sub[i].symbol != self.cluster_metal]]

        sub_sites, sub_plane_vector = self.get_sites(sub)
        return sub_sites, sub_plane_vector
    
    def get_deep_sites(self, atoms):
        deepList = self.get_deep_atoms(atoms)

        deep = atoms.copy()
        del deep[[i for i in range(len(deep)) if (i not in deepList) or deep[i].symbol != self.cluster_metal]]

        deep_sites, deep_plane_vector = self.get_sites(deep)

        return deep_sites, deep_plane_vector
    
    def get_sites(self, atoms):
        if len(atoms) == 1:
            sites = []
            for _ in range(2):
                sites.append([atoms.get_positions()[0][0],atoms.get_positions()[0][1],atoms.get_positions()[0][2], 1, 0])
            return np.array(sites), self._get_all_vector_list(top = [[0, 0, 1], [0, 0, 1]])
        elif len(atoms) == 2:
            sites = []
            top_vector, bridge_vector = [], []
            for atom in atoms:
                sites.append(np.append(atom.position, [1, 0]))
                top_vector.append([0, 0, 1])

            sites.append(np.array([(atoms.get_positions()[0][0] + atoms.get_positions()[1][0]) / 2,
                                   (atoms.get_positions()[0][1] + atoms.get_positions()[1][1]) / 2,
                                   (atoms.get_positions()[0][2] + atoms.get_positions()[1][2]) / 2,
                                   2, 0]))
            return np.array(sites), self._get_all_vector_list(top = [[0, 0, 1], [0,0,1]], bridge = [[0,0,1]])

        elif len(atoms) >= 3:
            atop = atoms.get_positions()
            pos_ext = atoms.get_positions()
            tri = Delaunay(pos_ext[:, :2])
            pos_nodes = pos_ext[tri.simplices]

            bridge_sites = []
            hollow_sites = []

            top_vector, bridge_vector, hollow_vector = [], [], []

            for _ in range(len(atop)):
                top_vector.append([0, 0, 1])

            for i in pos_nodes:
                if (self.distance(i[0][0], i[0][1], i[0][2], i[1][0], i[1][1], i[1][2])) < 3.0:
                    bridge_sites.append((i[0] + i[1]) / 2)
                    bridge_vector.append(self._get_normal_vector(i))
                else:
                    hollow_sites.append((i[0] + i[1]) / 2)
                    hollow_vector.append(self._get_normal_vector(i))
                if (self.distance(i[2][0], i[2][1], i[2][2], i[1][0], i[1][1], i[1][2])) < 3.0:
                    bridge_sites.append((i[2] + i[1]) / 2)
                    bridge_vector.append(self._get_normal_vector(i))
                else:
                    hollow_sites.append((i[2] + i[1]) / 2)
                    hollow_vector.append(self._get_normal_vector(i))
                if (self.distance(i[0][0], i[0][1], i[0][2], i[2][0], i[2][1], i[2][2])) < 3.0:
                    bridge_sites.append((i[0] + i[2]) / 2)
                    bridge_vector.append(self._get_normal_vector(i))
                else:
                    hollow_sites.append((i[0] + i[2]) / 2)
                    hollow_vector.append(self._get_normal_vector(i))

            top_sites = np.array(atop)
            hollow_sites = np.array(hollow_sites)
            bridge_sites = np.array(bridge_sites)

            sites_1 = []
            total_sites = []

            for i in top_sites:
                sites_1.append(np.transpose(np.append(i, 1)))
            for i in bridge_sites:
                sites_1.append(np.transpose(np.append(i, 2)))
            for i in hollow_sites:
                sites_1.append(np.transpose(np.append(i, 3)))
            for i in sites_1:
                total_sites.append(np.append(i, 0))

            total_sites = np.array(total_sites)
            total_plane_vector = self._get_all_vector_list(top_vector, bridge_vector, hollow_vector)

            return total_sites, total_plane_vector
        
    def _get_normal_vector(self, tri):
        A_B = np.cross(tri[1] - tri[0], tri[2] - tri[0])
        f_vector = A_B/np.linalg.norm(A_B)
        return f_vector

    def _get_all_vector_list(self, top = None, bridge = None, hollow = None):
        total_n_vector = []
        if top:
            for top_vector in top:
                total_n_vector.append(top_vector)
        if bridge:
            for bridge_vector in bridge:
                total_n_vector.append(bridge_vector)
        if hollow:
            for hollow_vector in hollow:
                total_n_vector.append(hollow_vector)
        return np.array(total_n_vector)
    
    def get_atom_info(self, atoms:ase.Atoms) -> ase.Atoms:
        layerList = self.get_layer_atoms(atoms)
        surfList = self.get_surf_atoms(atoms)
        subList = self.get_sub_atoms(atoms)
        deepList = self.get_deep_atoms(atoms)
        
        layer = atoms.copy()
        del layer[[i for i in range(len(layer)) if i not in layerList]]
        layer_atom = layer.get_positions()

        surf = atoms.copy()
        del surf[[i for i in range(len(surf)) if i not in surfList]]
        surf_atom = surf.get_positions()

        sub = atoms.copy()
        del sub[[i for i in range(len(sub)) if i not in subList]]
        sub_atom = sub.get_positions()

        deep = atoms.copy()
        del deep[[i for i in range(len(deep)) if i not in deepList]]
        deep_atom = deep.get_positions()

        return layer_atom, surf_atom, sub_atom, deep_atom

    '''------------This part will gain adsorbate information--------------'''
    def layer_O_atom_list(self, slab:ase.Atoms) -> List:
        slab = self._get_cluster(slab, with_zeolite=self.in_zeolite)
        layer_O = []
        layer_O_atom_list = []
        layer_OObond_list = []
        layer_List = self.get_layer_atoms(slab)

        for i in slab:
            if i.index in layer_List and i.symbol == 'O':
                layer_O.append(i.index)
        
        if layer_O:
            ana = Analysis(slab)
            OObonds = ana.get_bonds('O','O',unique = True)
            if OObonds[0]:
                for i in OObonds[0]:
                    if i[0] in layer_O or i[1] in layer_O:
                        layer_OObond_list.append(i[0])
                        layer_OObond_list.append(i[1])

            for j in layer_O:
                if j not in layer_OObond_list:
                    layer_O_atom_list.append(j)
        layer_O_atom_list = self._map_zeolite(layer_O_atom_list,with_zeolite=True)
        return layer_O_atom_list
    
    def sub_O_atom_list(self, slab:ase.Atoms) -> List:
        slab = self._get_cluster(slab, with_zeolite=self.in_zeolite)
        sub_O = []
        sub_O_atom_list = []
        sub_OObond_list = []
        sub_List = self.get_sub_atoms(slab)

        for i in slab:
            if i.index in sub_List and i.symbol == 'O':
                sub_O.append(i.index)
        
        if sub_O:
            ana = Analysis(slab)
            OObonds = ana.get_bonds('O','O',unique = True)
            if OObonds[0]:
                for i in OObonds[0]:
                    if i[0] in sub_O or i[1] in sub_O:
                        sub_OObond_list.append(i[0])
                        sub_OObond_list.append(i[1])

            for j in sub_O:
                if j not in sub_OObond_list:
                    sub_O_atom_list.append(j)
        sub_O_atom_list = self._map_zeolite(sub_O_atom_list, with_zeolite=True)
        return sub_O_atom_list
    
    def get_O_info(self, slab:ase.Atoms) -> List:
        slab = self._get_cluster(slab, with_zeolite=self.in_zeolite)

        layer_O_total = []
        sub_O_total = []

        total_O_list = []
        total_layer_atom_list = []
        total_sub_atom_list = []

        for atom in slab:
            if atom.symbol == 'O':
                total_O_list.append(atom.index)

        for facet in self.total_surfaces:
            slab= self.cluster_actions.cluster_rotation(slab, facet, self.center_point)
            layer_list = self.get_layer_atoms(slab)
            sub_list = self.get_sub_atoms(slab)

            for i in layer_list:
                if i not in total_layer_atom_list:
                    total_layer_atom_list.append(i)

            for i in sub_list:
                if i not in total_sub_atom_list:
                    total_sub_atom_list.append(i)

            slab = self.cluster_actions.recover_rotation(slab, facet,self.center_point)

        for j in total_layer_atom_list:
            if j in total_O_list:
                layer_O_total.append(j)
        
        for j in total_sub_atom_list:
            if j in total_O_list:
                sub_O_total.append(j)

        layer_O_total = self._map_zeolite(layer_O_total, with_zeolite=self.in_zeolite)
        sub_O_total = self._map_zeolite(sub_O_total, with_zeolite=self.in_zeolite)
        return layer_O_total, sub_O_total
    
    '''----------------Zeolite functions-------------------'''
    def _get_zeolite(self, system:ase.Atoms) -> ase.Atoms:
        if self.in_zeolite:
            return system[[a.index for a in system if a.index in range(len(self.zeolite))]]
        else:
            return None

    def _get_cluster(self, system:ase.Atoms, with_zeolite:Optional[bool]=False) -> ase.Atoms:
        if with_zeolite:
            return system[[a.index for a in system if a.index not in range(len(self.zeolite))]]
        else:
            return system
    
    def _get_system(self, atoms:ase.Atoms, with_zeolite:Optional[bool] = False) -> ase.Atoms:
        if with_zeolite:
            return self.zeolite + atoms
        else:
            return atoms
    
    def get_free_atoms(self, system: ase.Atoms) -> ase.Atoms:
        free_list = self._get_free_atoms_list(system)
        return system[free_list]
    
    def _get_free_atoms_list(self, system: ase.Atoms, center_point:List = None, d_max:float = 15.0) -> List:
        """
        The function is to get the free atoms on zeolite
        """
        central_cluster = self._get_cluster(system, with_zeolite=self.in_zeolite)
        if not center_point:
            center_point = self.cluster_actions.get_center_point(central_cluster)
        d_max = np.max(np.sqrt(np.diagonal(np.inner(central_cluster.get_positions() - center_point,
                                                    central_cluster.get_positions() - center_point)))) + 3.0


        print(f"The d max of cluster is {d_max}")
        d_array = np.sqrt(np.diagonal(np.inner(system.get_positions() - center_point,
                                               system.get_positions() - center_point)))
        free_list = np.where(d_array < d_max)[0].tolist()
        print(f"The length of free list is {len(free_list)}")
        return free_list
    
    def _map_zeolite(self, idx_list: List, with_zeolite: Optional[bool] = False) -> List:
        if with_zeolite:
            return (np.array(idx_list) + len(self.zeolite)).tolist()
        else:
            return idx_list
    
    '''---------------Neighbour facet----------------------'''
    def neighbour_facet(self, atoms:ase.Atoms, facet:List) -> List:
        surface_list = self.get_surf_atoms(atoms)
        atoms = self.cluster_actions.recover_rotation(atoms, facet, self.center_point)
        neighbour_facet = []
        neighbour_facet.append(facet)
        for selected_facet in self.total_surfaces:
            if selected_facet[0] != facet[0] or selected_facet[1] != facet[1] or selected_facet[2] != facet[2]:
                atoms = self.cluster_actions.cluster_rotation(atoms, selected_facet, self.center_point)
                selected_surface_list = self.get_surf_atoms(atoms)
                atoms = self.cluster_actions.recover_rotation(atoms, selected_facet, self.center_point)
                repeat_atoms = [i for i in selected_surface_list if i in surface_list]
                if len(repeat_atoms) >= 2:
                    neighbour_facet.append(selected_facet)
        return neighbour_facet
    
    def distance(self, x1, y1, z1, x2, y2, z2):
        dis = math.sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2) + (z1 - z2) * (z1 - z2))
        return dis
    
    def mid_point(self, slab:ase.Atoms, List) -> List[float]:
        sum_x = 0
        sum_y = 0
        sum_z = 0
        for i in slab:
            if i.index in List:
                sum_x += slab.get_positions()[i.index][0]
                sum_y += slab.get_positions()[i.index][1]
                sum_z += slab.get_positions()[i.index][2]
        mid_point = [sum_x/len(List), sum_y/len(List), sum_z/len(List)]
        return mid_point
    
    def to_get_min_distances(self, a, min_point):
        for i in range(len(a) - 1):
            for j in range(len(a)-i-1):
                if a[j] > a[j+1]:
                    a[j], a[j+1] = a[j+1], a[j]
        if len(a):
            if len(a) < min_point:
                return a[-1]
            else:
                return a[min_point - 1]
        else:
            return False
        
    def atom_to_traj_distance(self, atom_A, atom_B, atom_C):
        d_AB = self.distance(atom_A[0], atom_A[1], atom_A[2], atom_B[0], atom_B[1], atom_B[2])
        d = abs((atom_C[0]-atom_A[0])*(atom_A[0]-atom_B[0])+
                (atom_C[1]-atom_A[1])*(atom_A[1]-atom_B[1])+
                (atom_C[2]-atom_A[2])*(atom_A[2]-atom_B[2])) / d_AB
        return d
    
    '''----------------Whether can do such action------------------'''
    def _can_adsorb(self, atoms:ase.Atoms, target_site: np.ndarray, neigh_atom_index_list:List, ads_vector:List = None) -> bool:
        action_can_do = False
        d_list = []
        for atom_idx in neigh_atom_index_list:
            atom = atoms[atom_idx]
            if atom.symbol == 'O':
                d = self.distance(target_site[0], target_site[1], target_site[2] + 1.5,atom.position[0], atom.position[1], atom.position[2])
                d_list.append(d)
        
        if d_list:
            if min(d_list) > 2 * d_O_O:
                action_can_do = True
        else:
            action_can_do = True

        return action_can_do

    def _can_diffuse(self, atoms:ase.Atoms, neigh_atom_index_list) -> bool:
        action_can_do = False
        all_single_layer_O = self.layer_O_atom_list(atoms)
        for atom_idx in neigh_atom_index_list:
            atom = atoms[atom_idx]
            if atom.symbol == 'O' and atom_idx in all_single_layer_O:
                action_can_do = True
                break
        return action_can_do

    def _can_dissociate(self, atoms:ase.Atoms, neigh_atom_index_list) -> bool:
        action_can_do = False
        all_dissociate_list = self.get_dissociate_O2_list(atoms)
        for mole in all_dissociate_list:
            if mole[0] in neigh_atom_index_list and mole[1] in neigh_atom_index_list:
                action_can_do = True
                break
        return action_can_do
    
    def _can_desorb(self, atoms:ase.Atoms, neigh_atom_index_list:List) -> bool:
        action_can_do = False
        _,  all_desorb_list = self.to_desorb_adsorbate(atoms)
        for mole in all_desorb_list:
            if len(mole) == 2:
                if mole[0] in neigh_atom_index_list and mole[1] in neigh_atom_index_list:
                    action_can_do = True
                    break
            elif len(mole) == 3:
                if mole[0] in neigh_atom_index_list and mole[1] in neigh_atom_index_list and mole[2] in neigh_atom_index_list:
                    action_can_do = True
                    break
        return action_can_do
    
    '''---------The following part will be on the Transition state---------------'''
    def transition_state_search(self, previous_energy, current_energy, action):
        if action in [0,8]:
            barrier = current_energy - previous_energy + 0.4
        elif action == 1:
            if current_energy - previous_energy < -1.0:
                barrier = 0
            elif current_energy - previous_energy >= -1.0 and current_energy - previous_energy <= 1.0:
                barrier = np.random.normal(2, 2/3)
            else:
                barrier = 4.0

        elif action == 2 or action == 3:
            if current_energy-previous_energy < 0:
                barrier = 0
            else:
                barrier =  current_energy-previous_energy
        elif action == 4:
            barrier = 1.5
        elif action == 5:
            barrier = 0.4789 *(current_energy - previous_energy) + 0.8986
        elif action == 6:
            barrier = 0.6935 * (current_energy - previous_energy) + 0.6997
        elif action == 7:
            barrier = 0.65 + 0.84 * (current_energy - previous_energy)
        else:
            barrier = 1.5
                
        if barrier > 5.0:
            barrier = 5.0
        elif barrier < 0:
            barrier = 0

        return barrier
    
    def get_prob_list(self, prob_list:List[float]) -> List[float]:
        sum_prob = sum(prob_list)

        if sum_prob:
            for i in range(len(prob_list)):
                prob_list[i] = prob_list[i] / sum_prob

        normalized_prob_list = []
        for j in range(len(prob_list)):
            sum_k = 0
            for k in range(j + 1):
                sum_k += prob_list[k]
            normalized_prob_list.append(sum_k)
        
        normalized_prob_list.append(0.0)

        return normalized_prob_list
    

    '''---------------calculate single mole energy -------------------------'''
    def add_mole(self, mole:str) -> float:
        mole = molecule(mole)
        energy = self.calculator.to_calc(mole, calc_type = 'single')
        return energy