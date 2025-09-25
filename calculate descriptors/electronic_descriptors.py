"""
Module for calculating electronic descriptors for crystal structures
"""
import numpy as np
from pymatgen.core.periodic_table import Element
from pymatgen.core.structure import Structure
from pymatgen.analysis.local_env import CrystalNN
from scipy.spatial.distance import pdist, squareform

def calculate_average_electronegativity(structure):
    """
    Calculate average electronegativity of the structure.

    Args:
        structure (Structure): Pymatgen Structure object.

    Returns:
        float: Average electronegativity.
    """
    try:
        electronegativities = [Element(str(site.specie)).X for site in structure]
        return np.mean(electronegativities)
    except Exception as e:
        print(f"Error calculating electronegativity: {e}")
        return None

def calculate_electronegativity_difference(structure):
    """
    Calculate the maximum electronegativity difference in the structure.

    Args:
        structure (Structure): Pymatgen Structure object.

    Returns:
        float: Maximum electronegativity difference.
    """
    try:
        electronegativities = [Element(str(site.specie)).X for site in structure]
        return max(electronegativities) - min(electronegativities)
    except Exception as e:
        print(f"Error calculating electronegativity difference: {e}")
        return None

def calculate_average_polarizability(structure):
    """
    Calculate average atomic polarizability.
    
    Args:
        structure (Structure): Pymatgen Structure object.
        
    Returns:
        float: Average polarizability.
    """
    # Atomic polarizability values in cubic angstroms (approximate values)
    polarizability_dict = {
        'H': 0.667, 'He': 0.205, 'Li': 24.3, 'Be': 5.60, 'B': 3.03, 
        'C': 1.76, 'N': 1.10, 'O': 0.802, 'F': 0.557, 'Ne': 0.396,
        'Na': 24.1, 'Mg': 10.6, 'Al': 8.34, 'Si': 5.38, 'P': 3.63,
        'S': 2.90, 'Cl': 2.18, 'Ar': 1.64, 'K': 43.4, 'Ca': 22.8,
        'Zn': 7.1, 'Ga': 8.12, 'Ge': 6.07, 'As': 4.31, 'Se': 3.77,
        'Br': 3.05, 'Kr': 2.48, 'Rb': 47.3, 'Sr': 27.6, 'Zr': 17.9,
        'Ag': 7.2, 'Cd': 7.36, 'In': 10.2, 'Sn': 7.70, 'Sb': 6.60,
        'Te': 5.50, 'I': 5.35, 'Xe': 4.04, 'Cs': 59.6, 'Ba': 39.7
    }
    
    try:
        elements = [str(site.specie.symbol) for site in structure]
        polarizabilities = []
        
        for element in elements:
            if element in polarizability_dict:
                polarizabilities.append(polarizability_dict[element])
        
        if polarizabilities:
            return np.mean(polarizabilities)
        return None
    except Exception as e:
        print(f"Error calculating polarizability: {e}")
        return None

def calculate_bond_polarity(structure):
    """
    Calculate average bond polarity.
    
    Args:
        structure (Structure): Pymatgen Structure object.
        
    Returns:
        float: Average bond polarity.
    """
    try:
        nn = CrystalNN()
        bond_polarities = []
        
        for i, site in enumerate(structure):
            element_i = str(site.specie.symbol)
            en_i = Element(element_i).X
            
            neighbors = nn.get_nn_info(structure, i)
            for neighbor in neighbors:
                j = neighbor['site_index']
                element_j = str(structure[j].specie.symbol)
                en_j = Element(element_j).X
                
                bond_polarity = abs(en_i - en_j)
                bond_polarities.append(bond_polarity)
        
        if bond_polarities:
            return np.mean(bond_polarities)
        return None
    except Exception as e:
        print(f"Error calculating bond polarity: {e}")
        return None

def calculate_ionization_potentials(structure):
    """
    Calculate average and range of ionization potentials.
    
    Args:
        structure (Structure): Pymatgen Structure object.
        
    Returns:
        tuple: (average IP, IP range)
    """
    try:
        elements = [site.specie.symbol for site in structure]
        ips = []
        
        for element in elements:
            try:
                ip = Element(element).ionization_energy
                if ip is not None:
                    ips.append(ip)
            except:
                continue
        
        if ips:
            avg_ip = np.mean(ips)
            ip_range = max(ips) - min(ips)
            return avg_ip, ip_range
        
        return None, None
    except Exception as e:
        print(f"Error calculating ionization potentials: {e}")
        return None, None

def calculate_electron_affinity(structure):
    """
    Calculate average electron affinity.
    
    Args:
        structure (Structure): Pymatgen Structure object.
        
    Returns:
        float: Average electron affinity.
    """
    # Electron affinity values in eV (approximate values)
    ea_dict = {
        'H': 0.754, 'He': 0, 'Li': 0.618, 'Be': 0, 'B': 0.277, 
        'C': 1.263, 'N': 0, 'O': 1.461, 'F': 3.401, 'Ne': 0,
        'Na': 0.548, 'Mg': 0, 'Al': 0.441, 'Si': 1.385, 'P': 0.746,
        'S': 2.077, 'Cl': 3.617, 'Ar': 0, 'K': 0.501, 'Ca': 0.024,
        'Zn': 0, 'Ga': 0.43, 'Ge': 1.233, 'As': 0.81, 'Se': 2.021,
        'Br': 3.364, 'Kr': 0
    }
    
    try:
        elements = [site.specie.symbol for site in structure]
        eas = []
        
        for element in elements:
            if element in ea_dict:
                eas.append(ea_dict[element])
        
        if eas:
            return np.mean(eas)
        return None
    except Exception as e:
        print(f"Error calculating electron affinity: {e}")
        return None

def calculate_charge_distribution(structure):
    """
    Calculate a measure of charge distribution uniformity.
    
    Args:
        structure (Structure): Pymatgen Structure object.
        
    Returns:
        float: Standard deviation of partial charges (higher means less uniform).
    """
    try:
        # Estimate partial charges based on electronegativity differences
        # This is a simplified model; for accurate charges, DFT calculations would be needed
        
        elements = [site.specie.symbol for site in structure]
        electronegativities = [Element(element).X for element in elements]
        
        avg_en = np.mean(electronegativities)
        
        # Estimate partial charges as difference from average electronegativity
        partial_charges = [en - avg_en for en in electronegativities]
        
        # Calculate standard deviation as a measure of charge distribution uniformity
        charge_std = np.std(partial_charges)
        
        return charge_std
    except Exception as e:
        print(f"Error calculating charge distribution: {e}")
        return None

def calculate_metallic_character(structure):
    """
    Calculate a measure of metallic character.
    
    Args:
        structure (Structure): Pymatgen Structure object.
        
    Returns:
        float: Score between 0-1 (higher is more metallic).
    """
    try:
        metallic_elements = 0
        total_elements = len(structure)
        
        # Count metallic elements
        for site in structure:
            element = site.specie.symbol
            if Element(element).is_metal:
                metallic_elements += 1
        
        # Metallic character as proportion of metallic elements
        if total_elements > 0:
            return metallic_elements / total_elements
        return None
    except Exception as e:
        print(f"Error calculating metallic character: {e}")
        return None

def calculate_electronic_descriptors(structure):
    """
    Calculate all electronic descriptors.
    
    Args:
        structure (Structure): Pymatgen Structure object.
        
    Returns:
        dict: Combined electronic descriptors.
    """
    descriptors = {}
    
    try:
        # Original descriptors
        avg_en = calculate_average_electronegativity(structure)
        if avg_en:
            descriptors['average_electronegativity'] = round(avg_en, 3)
            
        en_diff = calculate_electronegativity_difference(structure)
        if en_diff:
            descriptors['electronegativity_difference'] = round(en_diff, 3)
            
        avg_pol = calculate_average_polarizability(structure)
        if avg_pol:
            descriptors['average_polarizability'] = round(avg_pol, 3)
            
        bond_pol = calculate_bond_polarity(structure)
        if bond_pol:
            descriptors['average_bond_polarity'] = round(bond_pol, 3)
        
        # Additional descriptors
        avg_ip, ip_range = calculate_ionization_potentials(structure)
        if avg_ip:
            descriptors['average_ionization_potential'] = round(avg_ip, 3)
        if ip_range:
            descriptors['ionization_potential_range'] = round(ip_range, 3)
            
        avg_ea = calculate_electron_affinity(structure)
        if avg_ea:
            descriptors['average_electron_affinity'] = round(avg_ea, 3)
            
        charge_dist = calculate_charge_distribution(structure)
        if charge_dist:
            descriptors['charge_distribution_uniformity'] = round(charge_dist, 3)
            
        metallic_char = calculate_metallic_character(structure)
        if metallic_char:
            descriptors['metallic_character'] = round(metallic_char, 3)
    
    except Exception as e:
        print(f"Error calculating electronic descriptors: {e}")
        
    return descriptors 