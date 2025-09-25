"""
Module for calculating geometric descriptors from CIF data
"""
import math
from pymatgen.core.structure import Structure


def calculate_geometric_descriptors(structure):
    """
    Calculate geometric descriptors from the crystal structure.

    Args:
        structure (Structure): Pymatgen Structure object.

    Returns:
        dict: Calculated geometric descriptors.
    """
    descriptors = {}

    try:
        # Extract cell parameters
        a, b, c = structure.lattice.abc
        alpha, beta, gamma = structure.lattice.angles

        # Validate cell parameters
        if a <= 0 or b <= 0 or c <= 0:
            raise ValueError("Cell lengths must be positive.")
        if alpha <= 0 or beta <= 0 or gamma <= 0:
            raise ValueError("Cell angles must be positive.")

        # Basic cell parameters
        descriptors.update({
            'cell_length_a': round(a, 3),
            'cell_length_b': round(b, 3),
            'cell_length_c': round(c, 3),
            'cell_angle_alpha': round(alpha, 3),
            'cell_angle_beta': round(beta, 3),
            'cell_angle_gamma': round(gamma, 3)
        })

        # Advanced geometric descriptors
        volume = structure.volume
        surface_area = calculate_cell_surface_area(a, b, c)
        aspect_ratio = max(a, b, c) / min(a, b, c) if min(a, b, c) > 0 else None
        sphericity = calculate_sphericity(a, b, c)
        compactness = calculate_compactness(volume, surface_area)

        descriptors.update({
            'cell_volume': round(volume, 3),
            'cell_surface_area': round(surface_area, 3),
            'aspect_ratio': round(aspect_ratio, 3) if aspect_ratio else None,
            'sphericity': round(sphericity, 3) if sphericity else None,
            'compactness': round(compactness, 3) if compactness else None,
        })

    except Exception as e:
        print(f"Warning: Error calculating geometric descriptors - {str(e)}")
        for key in ['cell_length_a', 'cell_length_b', 'cell_length_c',
                    'cell_angle_alpha', 'cell_angle_beta', 'cell_angle_gamma',
                    'cell_volume', 'cell_surface_area', 'aspect_ratio',
                    'sphericity', 'compactness']:
            descriptors[key] = None

    return descriptors


def calculate_cell_surface_area(a, b, c):
    """
    Calculate the surface area of the unit cell.

    Args:
        a (float): Length of the first cell axis.
        b (float): Length of the second cell axis.
        c (float): Length of the third cell axis.

    Returns:
        float: Surface area of the unit cell.
    """
    return 2 * (a * b + b * c + a * c)


def calculate_sphericity(a, b, c):
    """
    Calculate sphericity (0-1, 1 being perfectly spherical).

    Args:
        a (float): Length of the first cell axis.
        b (float): Length of the second cell axis.
        c (float): Length of the third cell axis.

    Returns:
        float: Sphericity of the unit cell.
    """
    max_dim = max(a, b, c)
    min_dim = min(a, b, c)
    return min_dim / max_dim if max_dim > 0 else None


def calculate_compactness(volume, surface_area):
    """
    Calculate compactness (volume / surface_area^(3/2)).

    Args:
        volume (float): Volume of the unit cell.
        surface_area (float): Surface area of the unit cell.

    Returns:
        float: Compactness of the unit cell.
    """
    return volume / (surface_area ** (3/2)) if surface_area > 0 else None