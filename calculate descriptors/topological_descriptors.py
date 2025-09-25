"""
Module for calculating topological descriptors from CIF data
"""
import math
from pymatgen.core.structure import Structure
from pymatgen.core.periodic_table import Element
from pymatgen.analysis.local_env import VoronoiNN, CrystalNN
import networkx as nx


def calculate_coordination_numbers(structure):
    """
    Calculate coordination numbers and average degree.

    Args:
        structure (Structure): Pymatgen Structure object.

    Returns:
        dict: Coordination-related descriptors.
    """
    descriptors = {
        'average_coordination': None,
        'coordination_numbers': {},
        'average_degree': None
    }

    try:
        cnn = CrystalNN()
        coordination_sums = {}
        element_counts = {}
        total_coordination = 0
        count = 0

        for site in structure:
            element = str(site.specie)
            coordination = cnn.get_cn(structure, structure.index(site))
            coordination_sums[element] = coordination_sums.get(element, 0) + coordination
            element_counts[element] = element_counts.get(element, 0) + 1
            total_coordination += coordination
            count += 1

        if count > 0:
            descriptors['average_coordination'] = round(total_coordination / count, 3)
            descriptors['average_degree'] = round(total_coordination / count, 3)
            descriptors['coordination_numbers'] = {
                element: round(coordination_sums[element] / element_counts[element], 3)
                for element in element_counts
            }

    except Exception as e:
        print(f"Error calculating coordination numbers: {e}")

    return descriptors


def calculate_hydrogen_bond_descriptors(structure):
    """
    Calculate hydrogen bond donors and acceptors.

    Args:
        structure (Structure): Pymatgen Structure object.

    Returns:
        dict: Hydrogen bond descriptors.
    """
    descriptors = {
        'num_hbond_donors': 0,
        'num_hbond_acceptors': 0
    }
    
    try:
        cnn = CrystalNN()
        for site in structure:
            element = str(site.specie)
            if element in ['O', 'N']:
                descriptors['num_hbond_acceptors'] += 1
                neighbors = cnn.get_nn(structure, structure.index(site))
                for neighbor in neighbors:
                    if str(neighbor.specie) == 'H':
                        descriptors['num_hbond_donors'] += 1
            elif element == 'H':
                descriptors['num_hbond_donors'] += 1

    except Exception as e:
        print(f"Error calculating hydrogen bond descriptors: {e}")

    return descriptors


def calculate_topological_descriptors(structure):
    """
    Calculate all topological descriptors.

    Args:
        structure (Structure): Pymatgen Structure object.

    Returns:
        dict: Combined topological descriptors.
    """
    descriptors = {}
    
    try:
        descriptors.update(calculate_hydrogen_bond_descriptors(structure))
        descriptors.update(calculate_coordination_numbers(structure))
        
        # Clean None values
        descriptors = {k: v for k, v in descriptors.items() if v is not None}
        
    except Exception as e:
        print(f"Error calculating topological descriptors: {e}")

    return descriptors