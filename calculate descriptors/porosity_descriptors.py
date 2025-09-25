"""
Module for calculating porosity and surface descriptors for crystal structures
"""
import numpy as np
from math import pi
from pymatgen.core.structure import Structure
from pymatgen.core.periodic_table import Element
from scipy.spatial import Voronoi, ConvexHull
from scipy.spatial.distance import pdist, squareform

def calculate_accessible_volume(structure, probe_radius=1.2):
    """
    Calculate the accessible volume using a probe of given radius.
    Probe radius of 1.2 Å is typical for N2 gas.
    
    Args:
        structure (Structure): Pymatgen Structure object.
        probe_radius (float): Radius of the probe molecule in Å.
        
    Returns:
        float: Accessible volume fraction.
    """
    try:
        # Get the total volume
        total_volume = structure.volume
        
        # Calculate the volume occupied by atoms including the probe radius
        occupied_volume = 0
        for site in structure:
            element = str(site.specie.symbol)
            # Use atomic radius + probe radius
            radius = Element(element).atomic_radius + probe_radius
            # Calculate sphere volume
            site_volume = (4/3) * pi * (radius**3)
            occupied_volume += site_volume
        
        # Calculate accessible volume (approximate)
        # Note: This is a simplified approach. For more accurate results,
        # specialized tools like Zeo++ or CrystalExplorer would be needed.
        accessible_volume = max(0, total_volume - occupied_volume)
        accessible_fraction = accessible_volume / total_volume
        
        return accessible_fraction
    except Exception as e:
        print(f"Error calculating accessible volume: {e}")
        return None

def calculate_pore_dimensionality(structure, probe_radius=1.2, min_channel_width=2.8):
    """
    Estimate pore dimensionality (0D, 1D, 2D, 3D) based on structure analysis.
    This is a simplified implementation, as accurate pore dimensionality requires
    specialized algorithms.
    
    Args:
        structure (Structure): Pymatgen Structure object.
        probe_radius (float): Radius of the probe molecule in Å.
        min_channel_width (float): Minimum width to consider a channel in Å.
        
    Returns:
        int: Estimated pore dimensionality (0, 1, 2, or 3).
    """
    try:
        # This is a simplified approach based on accessible space in each crystallographic direction
        a, b, c = structure.lattice.abc
        
        # Check free space in each direction (simplistic approach)
        has_channel_a = has_channel_in_direction(structure, [1, 0, 0], min_channel_width, probe_radius)
        has_channel_b = has_channel_in_direction(structure, [0, 1, 0], min_channel_width, probe_radius)
        has_channel_c = has_channel_in_direction(structure, [0, 0, 1], min_channel_width, probe_radius)
        
        # Count dimensions with channels
        dimensionality = sum([has_channel_a, has_channel_b, has_channel_c])
        return dimensionality
    except Exception as e:
        print(f"Error calculating pore dimensionality: {e}")
        return None

def has_channel_in_direction(structure, direction, min_width, probe_radius):
    """
    Helper function to check if there's a channel in the given direction.
    
    Args:
        structure (Structure): Pymatgen Structure object.
        direction (list): Direction vector [a, b, c].
        min_width (float): Minimum width to consider a channel in Å.
        probe_radius (float): Radius of the probe molecule in Å.
        
    Returns:
        bool: True if a channel exists in the given direction.
    """
    # This is a simplified check - a proper implementation would use
    # a grid-based approach to check for continuous paths
    
    # Get the lattice vector in the specified direction
    lattice_vector = structure.lattice.get_cartesian_coords(direction)
    vector_length = np.linalg.norm(lattice_vector)
    unit_vector = lattice_vector / vector_length
    
    # Simple approach: check if there's a gap of min_width along the direction
    # This is a very simplified approach and may not be accurate for complex structures
    gaps = []
    for i in range(100):  # Sample 100 points along the direction
        point = unit_vector * (i / 100) * vector_length
        distance_to_nearest_atom = float('inf')
        
        for site in structure:
            site_coords = site.coords
            site_vector = site_coords - point
            distance = np.linalg.norm(site_vector)
            distance_to_nearest_atom = min(distance_to_nearest_atom, distance)
        
        if distance_to_nearest_atom > min_width:
            gaps.append(distance_to_nearest_atom)
    
    # If there are enough consecutive points with sufficient distance to atoms,
    # we consider this a channel
    has_channel = len(gaps) >= 5  # Arbitrary threshold
    return has_channel

def calculate_surface_area(structure, probe_radius=1.2):
    """
    Estimate the surface area accessible to a probe.
    Note: This is a simplified approach. For more accurate results,
    specialized tools should be used.
    
    Args:
        structure (Structure): Pymatgen Structure object.
        probe_radius (float): Radius of the probe molecule in Å.
        
    Returns:
        float: Estimated accessible surface area in Å².
    """
    try:
        # This is a very simplified approximation based on atom contributions
        surface_area = 0
        
        for site in structure:
            element = str(site.specie.symbol)
            radius = Element(element).atomic_radius + probe_radius
            # Surface area of a sphere
            site_surface = 4 * pi * (radius**2)
            
            # Adjust for overlap (very simplistic approach)
            # In reality, need to account for accessible surface area
            overlap_factor = 0.65  # Arbitrary reduction factor for overlap
            surface_area += site_surface * overlap_factor
            
        return surface_area
    except Exception as e:
        print(f"Error calculating surface area: {e}")
        return None

def calculate_pore_size_distribution(structure, probe_radius=1.2, num_samples=1000):
    """
    Estimate the pore size distribution.
    
    Args:
        structure (Structure): Pymatgen Structure object.
        probe_radius (float): Radius of the probe molecule in Å.
        num_samples (int): Number of random points to sample.
        
    Returns:
        tuple: (average_pore_diameter, max_pore_diameter)
    """
    try:
        # Get structure bounds
        bounds = structure.lattice.matrix
        bound_x = np.linalg.norm(bounds[0])
        bound_y = np.linalg.norm(bounds[1])
        bound_z = np.linalg.norm(bounds[2])
        
        # Random sampling of points in the unit cell
        distances = []
        for _ in range(num_samples):
            # Random point within unit cell
            rand_point = np.array([
                np.random.random() * bound_x,
                np.random.random() * bound_y,
                np.random.random() * bound_z
            ])
            
            # Find distance to nearest atom
            min_distance = float('inf')
            for site in structure:
                site_coords = site.coords
                distance = np.linalg.norm(site_coords - rand_point)
                adj_distance = distance - Element(str(site.specie.symbol)).atomic_radius
                min_distance = min(min_distance, adj_distance)
            
            if min_distance > 0:  # Point is not inside an atom
                distances.append(min_distance)
        
        # Calculate pore size statistics
        if distances:
            avg_pore_diameter = 2 * np.mean(distances)
            max_pore_diameter = 2 * max(distances)
            return avg_pore_diameter, max_pore_diameter
        
        return None, None
    except Exception as e:
        print(f"Error calculating pore size distribution: {e}")
        return None, None

def calculate_void_fraction(structure):
    """
    Calculate the void fraction of the structure.
    
    Args:
        structure (Structure): Pymatgen Structure object.
        
    Returns:
        float: Void fraction.
    """
    try:
        # Calculate total cell volume
        cell_volume = structure.volume
        
        # Calculate volume occupied by atoms
        occupied_volume = 0
        for site in structure:
            element = str(site.specie.symbol)
            radius = Element(element).atomic_radius
            if radius:
                atom_volume = (4/3) * pi * (radius ** 3)
                occupied_volume += atom_volume
        
        # Calculate void fraction
        void_fraction = 1.0 - (occupied_volume / cell_volume)
        return max(0.0, void_fraction)  # Ensure non-negative
    except Exception as e:
        print(f"Error calculating void fraction: {e}")
        return None

def calculate_specific_surface_area(structure, probe_radius=1.2):
    """
    Calculate specific surface area (surface area per unit mass).
    
    Args:
        structure (Structure): Pymatgen Structure object.
        probe_radius (float): Radius of the probe molecule in Å.
        
    Returns:
        float: Specific surface area in m²/g.
    """
    try:
        # Calculate surface area in Å²
        surface_area = calculate_surface_area(structure, probe_radius)
        
        if not surface_area:
            return None
        
        # Calculate mass in atomic mass units
        total_mass = 0
        for site in structure:
            element = str(site.specie.symbol)
            mass = Element(element).atomic_mass
            total_mass += mass
        
        # Convert to m²/g
        # 1 Å² = 1e-20 m²
        # mass is in amu, so we convert to g by dividing by Avogadro's number and multiplying by g/mol
        avogadro = 6.022e23
        specific_area = (surface_area * 1e-20) / (total_mass / avogadro)
        
        return specific_area
    except Exception as e:
        print(f"Error calculating specific surface area: {e}")
        return None

def calculate_porosity_descriptors(structure, probe_radius=1.2):
    """
    Calculate all porosity and surface descriptors.
    
    Args:
        structure (Structure): Pymatgen Structure object.
        probe_radius (float): Radius of the probe molecule in Å.
        
    Returns:
        dict: Combined porosity descriptors.
    """
    descriptors = {}
    
    try:
        # Original descriptors
        accessible_fraction = calculate_accessible_volume(structure, probe_radius)
        if accessible_fraction is not None:
            descriptors['accessible_volume_fraction'] = round(accessible_fraction, 3)
        
        pore_dim = calculate_pore_dimensionality(structure, probe_radius)
        if pore_dim is not None:
            descriptors['pore_dimensionality'] = pore_dim
        
        surface_area = calculate_surface_area(structure, probe_radius)
        if surface_area is not None:
            descriptors['accessible_surface_area'] = round(surface_area, 3)
        
        if accessible_fraction is not None:
            pore_volume = accessible_fraction * structure.volume
            descriptors['pore_volume'] = round(pore_volume, 3)
        
        # New descriptors
        avg_pore_diam, max_pore_diam = calculate_pore_size_distribution(structure, probe_radius)
        if avg_pore_diam:
            descriptors['average_pore_diameter'] = round(avg_pore_diam, 3)
        if max_pore_diam:
            descriptors['maximum_pore_diameter'] = round(max_pore_diam, 3)
            
        void_fraction = calculate_void_fraction(structure)
        if void_fraction is not None:
            descriptors['void_fraction'] = round(void_fraction, 3)
            
        specific_area = calculate_specific_surface_area(structure, probe_radius)
        if specific_area:
            descriptors['specific_surface_area_m2_g'] = round(specific_area, 3)
        
    except Exception as e:
        print(f"Error calculating porosity descriptors: {e}")
        
    return descriptors 