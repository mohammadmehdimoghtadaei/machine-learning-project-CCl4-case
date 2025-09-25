"""
Module for calculating crystal descriptors from CIF data.
"""
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

def calculate_crystal_descriptors(structure):
    """
    Determine the crystal system and space group.

    Args:
        structure (Structure): Pymatgen Structure object.

    Returns:
        dict: Crystal descriptors.
    """
    descriptors = {}

    try:
        analyzer = SpacegroupAnalyzer(structure)
        space_group = analyzer.get_space_group_symbol()
        crystal_system = analyzer.get_crystal_system()
        point_group = analyzer.get_point_group_symbol()

        descriptors.update({
            'space_group': space_group,
            'crystal_system': crystal_system,
            'point_group': point_group
        })
        
    except Exception as e:
        print(f"Warning: Error calculating crystal descriptors - {str(e)}")
        descriptors.update({
            'space_group': None,
            'crystal_system': None,
            'point_group': None
        })

    return descriptors