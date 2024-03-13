from dataclasses import dataclass
from typing import List, Dict, Tuple

import math
from math import cos,sin
import numpy as np

import ase

@dataclass
class ClusterActions():
    metal_ele:str = 'Pd'

    def cluster_rotation(self, atoms:ase.Atoms, facet:List, center_point:List = None) -> ase.Atoms:
        atoms = self.put_atoms_to_zero_point(atoms)
        if facet[0] == 0 and facet[1] == 0 and facet[2] == 1:
            atoms = self.rectify_atoms_positions(atoms, center_point)
            return atoms
        elif facet[0] == 0 and facet[1] == 0 and facet[2] == -1:
            zeta = math.acos(facet[2]/math.sqrt(facet[0] * facet[0] + facet[1] * facet[1]
                                                + facet[2] * facet[2]))
            facet = np.dot(self.matrix_y(zeta), facet)
            state = atoms.copy()
            for atom in state.positions:
                atom += np.array((np.dot(self.matrix_y(zeta),
                                        np.array(atom.tolist()).T).T).tolist()) - atom
            atoms.positions = state.get_positions()
            
            atoms = self.rectify_atoms_positions(atoms, center_point)
            return atoms
        else:
            zeta_1 = math.acos(facet[0]/math.sqrt(facet[0] * facet[0] +
                                                        facet[1] * facet[1]))
            state_1 = atoms.copy()
            if facet[1] > 0:
                facet_1 = np.dot(self.matrix_z(-zeta_1), facet)
                
                for atom in state_1.positions:
                    atom += np.array((np.dot(self.matrix_z(-zeta_1),
                                            np.array(atom.tolist()).T).T).tolist()) - atom
                
            else:
                facet_1 = np.dot(self.matrix_z(zeta_1), facet)
                for atom in state_1.positions:
                    atom += np.array((np.dot(self.matrix_z(zeta_1),
                                            np.array(atom.tolist()).T).T).tolist()) - atom
            
            atoms.positions = state_1.get_positions()
            zeta_2 = math.acos(facet_1[2]/math.sqrt(facet_1[0] * facet_1[0] +
                                                    facet_1[2] * facet_1[2]))
            facet_2 = np.dot(self.matrix_y(zeta_2), facet_1)
            state_2 = atoms.copy()
            for atom in state_2.positions:
                atom += np.array((np.dot(self.matrix_y(zeta_2),
                                        np.array(atom.tolist()).T).T).tolist()) - atom
            atoms.positions = state_2.get_positions()
            atoms = self.rectify_atoms_positions(atoms, center_point)
            return atoms

    def recover_rotation(self, atoms:ase.Atoms, facet:List, center_point:List = None) -> ase.Atoms:
        atoms = self.put_atoms_to_zero_point(atoms)
        if facet[0] == 0 and facet[1] == 0 and facet[2] == 1:
            atoms = self.rectify_atoms_positions(atoms, center_point)
            return atoms
        elif facet[0] == 0 and facet[1] == 0 and facet[2] == -1:
            zeta = math.acos(facet[2]/math.sqrt(facet[0] * facet[0] + facet[1] * facet[1]
                                                + facet[2] * facet[2]))
            facet = np.dot(self.matrix_y(-zeta), facet)
            state = atoms.copy()
            for atom in state.positions:
                atom += np.array((np.dot(self.matrix_y(-zeta),
                                        np.array(atom.tolist()).T).T).tolist()) - atom
            atoms.positions = state.get_positions()

            atoms = self.rectify_atoms_positions(atoms, center_point)
                
            return atoms
        else:
            zeta_1 = math.acos(facet[0]/math.sqrt(facet[0] * facet[0] +
                                                    facet[1] * facet[1]))

            if facet[1] > 0:
                facet_1 = np.dot(self.matrix_z(-zeta_1), facet)
            else:
                facet_1 = np.dot(self.matrix_z(zeta_1), facet)
                
            zeta_2 = math.acos(facet_1[2]/math.sqrt(facet_1[0] * facet_1[0] +
                                                    facet_1[2] * facet_1[2]))
            facet_2 = np.dot(self.matrix_y(zeta_2), facet_1)

            state_1 = atoms.copy()
            for atom in state_1.positions:
                atom += np.array((np.dot(self.matrix_y(-zeta_2),
                                        np.array(atom.tolist()).T).T).tolist()) - atom
            atoms.positions = state_1.get_positions()

            state_2 = atoms.copy()
            if facet[1] > 0:
                for atom in state_2.positions:
                    atom += np.array((np.dot(self.matrix_z(zeta_1),
                                            np.array(atom.tolist()).T).T).tolist()) - atom
            else:
                for atom in state_2.positions:
                    atom += np.array((np.dot(self.matrix_z(-zeta_1),
                                            np.array(atom.tolist()).T).T).tolist()) - atom
            atoms.positions = state_2.get_positions()

            atoms = self.rectify_atoms_positions(atoms, center_point)

            return atoms
        
    def rectify_atoms_positions(self, atoms: ase.Atoms, center_point:List = None) -> ase.Atoms:   # put atom to center point
        current_center_point = self.get_center_point(atoms)
        if center_point:
            det = np.array([center_point[0]- current_center_point[0], 
                            center_point[1] - current_center_point[1], 
                            center_point[2] - current_center_point[2]])
        else:
            det = np.array([atoms.get_cell()[0][0]/2 - current_center_point[0], 
                            atoms.get_cell()[1][1]/2 - current_center_point[1], 
                            atoms.get_cell()[2][2]/2 - current_center_point[2]])

        for position in atoms.positions:
            position += det

        return atoms
    
    def get_center_point(self, atoms: ase.Atoms) -> List[str]:
        sum_x = 0
        sum_y = 0
        sum_z = 0

        n_Pd = 0

        for atom in atoms:
            if atom.symbol == self.metal_ele:
                sum_x += atom.position[0]
                sum_y += atom.position[1]
                sum_z += atom.position[2]
                n_Pd += 1

        # return [sum_x/len(atoms.get_positions()), sum_y/len(atoms.get_positions()), sum_z/len(atoms.get_positions())]
        return [sum_x/n_Pd, sum_y/n_Pd, sum_z/n_Pd]
    
    def put_atoms_to_zero_point(self, atoms):
        current_center_point = self.get_center_point(atoms)

        det = np.array(current_center_point)

        for position in atoms.positions:
            position += -det

        return atoms
    
    def matrix_x(self, zeta):
        return np.array([[1, 0, 0],
                     [0, cos(zeta), sin(zeta)],
                      [0, -sin(zeta), cos(zeta)]
                      ])
    def matrix_y(self, zeta):
        return np.array([[cos(zeta), 0, -sin(zeta)],
                        [0, 1, 0],
                        [sin(zeta), 0, cos(zeta)]])
    def matrix_z(self, zeta):
        return np.array([[cos(zeta), -sin(zeta), 0],
                        [sin(zeta), cos(zeta), 0],
                        [0, 0, 1]])