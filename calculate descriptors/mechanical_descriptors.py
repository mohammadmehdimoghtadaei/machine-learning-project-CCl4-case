"""
Module for calculating mechanical descriptors for crystal structures
"""
import numpy as np
from pymatgen.core.structure import Structure
from pymatgen.core.periodic_table import Element
from pymatgen.analysis.local_env import CrystalNN

def calculate_packing_efficiency(structure):
    """
    Calculate the packing efficiency of the crystal structure.
    
    Args:
        structure (Structure): Pymatgen Structure object.
        
    Returns:
        float: Packing efficiency as a fraction.
    """
    try:
        # Calculate the total volume of the unit cell
        cell_volume = structure.volume
        
        # Calculate the total volume of atoms (using atomic radii)
        atomic_volume = 0
        for site in structure:
            element = str(site.specie.symbol)
            radius = Element(element).atomic_radius
            if radius:
                # Volume of a sphere
                atom_volume = (4/3) * np.pi * (radius ** 3)
                atomic_volume += atom_volume
        
        # Calculate packing efficiency
        packing_efficiency = atomic_volume / cell_volume
        return packing_efficiency
    except Exception as e:
        print(f"Error calculating packing efficiency: {e}")
        return None

def estimate_bulk_modulus(structure):
    """
    Estimate the bulk modulus using a simple model based on 
    bond strengths and connectivity.
    
    Args:
        structure (Structure): Pymatgen Structure object.
        
    Returns:
        float: Estimated bulk modulus in GPa.
    """
    try:
        # This is a simplified approach that correlates 
        # packing efficiency with bulk modulus
        packing_eff = calculate_packing_efficiency(structure)
        
        # Calculate average bond strength
        nn = CrystalNN()
        bond_strength_sum = 0
        bond_count = 0
        
        for i, site in enumerate(structure):
            element_i = str(site.specie.symbol)
            neighbors = nn.get_nn_info(structure, i)
            
            for neighbor in neighbors:
                j = neighbor['site_index']
                element_j = str(structure[j].specie.symbol)
                
                # Very simplified bond strength estimation based on electronegativity difference
                en_i = Element(element_i).X
                en_j = Element(element_j).X
                bond_strength = 1 / (1 + abs(en_i - en_j))  # Simplified model
                
                bond_strength_sum += bond_strength
                bond_count += 1
        
        avg_bond_strength = bond_strength_sum / bond_count if bond_count > 0 else 0
        
        # Simplified model for bulk modulus estimation
        # This is a very rough approximation and should be validated
        est_bulk_modulus = 200 * packing_eff * avg_bond_strength
        
        return est_bulk_modulus
    except Exception as e:
        print(f"Error estimating bulk modulus: {e}")
        return None

def estimate_shear_modulus(structure):
    """
    Estimate the shear modulus based on the crystal structure.
    
    Args:
        structure (Structure): Pymatgen Structure object.
        
    Returns:
        float: Estimated shear modulus in GPa.
    """
    try:
        # Estimate based on bulk modulus (simplified relationship)
        bulk_modulus = estimate_bulk_modulus(structure)
        if bulk_modulus is None:
            return None
            
        # For most materials, shear modulus is roughly 0.3-0.5 of bulk modulus
        # This is a very rough approximation
        shear_factor = 0.4  # Approximate factor
        shear_modulus = bulk_modulus * shear_factor
        
        return shear_modulus
    except Exception as e:
        print(f"Error estimating shear modulus: {e}")
        return None

def estimate_youngs_modulus(structure):
    """
    Estimate Young's modulus based on estimated bulk and shear moduli.
    
    Args:
        structure (Structure): Pymatgen Structure object.
        
    Returns:
        float: Estimated Young's modulus in GPa.
    """
    try:
        bulk_modulus = estimate_bulk_modulus(structure)
        shear_modulus = estimate_shear_modulus(structure)
        
        if bulk_modulus is None or shear_modulus is None:
            return None
            
        # Relationship between Young's modulus, bulk modulus, and shear modulus
        youngs_modulus = (9 * bulk_modulus * shear_modulus) / (3 * bulk_modulus + shear_modulus)
        
        return youngs_modulus
    except Exception as e:
        print(f"Error estimating Young's modulus: {e}")
        return None

def estimate_poisson_ratio(structure):
    """
    Estimate Poisson's ratio based on estimated bulk and shear moduli.
    
    Args:
        structure (Structure): Pymatgen Structure object.
        
    Returns:
        float: Estimated Poisson's ratio (dimensionless).
    """
    try:
        bulk_modulus = estimate_bulk_modulus(structure)
        shear_modulus = estimate_shear_modulus(structure)
        
        if bulk_modulus is None or shear_modulus is None:
            return None
            
        # Relationship between Poisson's ratio, bulk modulus, and shear modulus
        poisson_ratio = (3 * bulk_modulus - 2 * shear_modulus) / (2 * (3 * bulk_modulus + shear_modulus))
        
        return poisson_ratio
    except Exception as e:
        print(f"Error estimating Poisson's ratio: {e}")
        return None

def calculate_average_bond_length(structure):
    """
    Calculate the average bond length in the structure.
    
    Args:
        structure (Structure): Pymatgen Structure object.
        
    Returns:
        float: Average bond length in Å.
    """
    try:
        nn = CrystalNN()
        bond_lengths = []
        
        for i, site in enumerate(structure):
            neighbors = nn.get_nn_info(structure, i)
            for neighbor in neighbors:
                # Get the distance directly from the site positions
                j = neighbor['site_index']
                site_j = structure[j]
                
                # Calculate distance between sites
                distance = site.distance(site_j)
                bond_lengths.append(distance)
        
        if bond_lengths:
            return np.mean(bond_lengths)
        return None
    except Exception as e:
        print(f"Error calculating average bond length: {e}")
        return None

def estimate_thermal_expansion(structure):
    """
    Estimate thermal expansion coefficient based on bond strengths.
    
    Args:
        structure (Structure): Pymatgen Structure object.
        
    Returns:
        float: Estimated thermal expansion coefficient in 10^-6 K^-1.
    """
    try:
        # Calculate average bond strength (inversely related to thermal expansion)
        nn = CrystalNN()
        bond_strength_sum = 0
        bond_count = 0
        
        for i, site in enumerate(structure):
            element_i = str(site.specie.symbol)
            neighbors = nn.get_nn_info(structure, i)
            
            for neighbor in neighbors:
                j = neighbor['site_index']
                element_j = str(structure[j].specie.symbol)
                
                # Simplified bond strength calculation
                en_i = Element(element_i).X
                en_j = Element(element_j).X
                bond_strength = 1 / (1 + abs(en_i - en_j))
                
                bond_strength_sum += bond_strength
                bond_count += 1
        
        if bond_count == 0:
            return None
            
        avg_bond_strength = bond_strength_sum / bond_count
        
        # Empirical inverse relationship between bond strength and thermal expansion
        # This is a very rough approximation
        thermal_expansion = 20 / (1 + 5 * avg_bond_strength)
        
        return thermal_expansion
    except Exception as e:
        print(f"Error estimating thermal expansion: {e}")
        return None

def estimate_hardness(structure):
    """
    Estimate hardness based on bond strengths and connectivity.
    
    Args:
        structure (Structure): Pymatgen Structure object.
        
    Returns:
        float: Estimated hardness in Mohs scale (approximated).
    """
    try:
        # Get bulk modulus (related to hardness)
        bulk_modulus = estimate_bulk_modulus(structure)
        if bulk_modulus is None:
            return None
            
        # Empirical relationship between bulk modulus and Mohs hardness
        # Based on very rough approximation
        mohs_hardness = 0.0002 * bulk_modulus + 1
        
        # Clamp between 1-10 (Mohs scale)
        mohs_hardness = max(1, min(10, mohs_hardness))
        
        return mohs_hardness
    except Exception as e:
        print(f"Error estimating hardness: {e}")
        return None

def calculate_anisotropy_index(structure):
    """
    Calculate a simplified anisotropy index based on lattice parameters.
    
    Args:
        structure (Structure): Pymatgen Structure object.
        
    Returns:
        float: Anisotropy index (1 = isotropic, >1 = anisotropic).
    """
    try:
        a, b, c = structure.lattice.abc
        max_param = max(a, b, c)
        min_param = min(a, b, c)
        
        # Simple ratio as anisotropy measure
        anisotropy = max_param / min_param if min_param > 0 else None
        
        return anisotropy
    except Exception as e:
        print(f"Error calculating anisotropy index: {e}")
        return None

def calculate_mechanical_descriptors(structure):
    """
    Calculate all mechanical descriptors.
    
    Args:
        structure (Structure): Pymatgen Structure object.
        
    Returns:
        dict: Combined mechanical descriptors.
    """
    descriptors = {}
    
    try:
        # Original descriptors
        packing_eff = calculate_packing_efficiency(structure)
        if packing_eff is not None:
            descriptors['packing_efficiency'] = round(packing_eff, 3)
        
        bulk_modulus = estimate_bulk_modulus(structure)
        if bulk_modulus is not None:
            descriptors['estimated_bulk_modulus_GPa'] = round(bulk_modulus, 3)
        
        shear_modulus = estimate_shear_modulus(structure)
        if shear_modulus is not None:
            descriptors['estimated_shear_modulus_GPa'] = round(shear_modulus, 3)
        
        youngs_modulus = estimate_youngs_modulus(structure)
        if youngs_modulus is not None:
            descriptors['estimated_youngs_modulus_GPa'] = round(youngs_modulus, 3)
        
        poisson_ratio = estimate_poisson_ratio(structure)
        if poisson_ratio is not None:
            descriptors['estimated_poisson_ratio'] = round(poisson_ratio, 3)
            
        # Additional descriptors
        avg_bond_length = calculate_average_bond_length(structure)
        if avg_bond_length is not None:
            descriptors['average_bond_length'] = round(avg_bond_length, 3)
            
        thermal_expansion = estimate_thermal_expansion(structure)
        if thermal_expansion is not None:
            descriptors['estimated_thermal_expansion'] = round(thermal_expansion, 3)
            
        hardness = estimate_hardness(structure)
        if hardness is not None:
            descriptors['estimated_mohs_hardness'] = round(hardness, 3)
            
        anisotropy = calculate_anisotropy_index(structure)
        if anisotropy is not None:
            descriptors['anisotropy_index'] = round(anisotropy, 3)
        
    except Exception as e:
        print(f"Error calculating mechanical descriptors: {e}")
        
    return descriptors 