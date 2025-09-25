import math
import numpy as np
from functools import lru_cache
from collections import Counter
from pymatgen.core.periodic_table import Element
from pymatgen.core.structure import Structure
from pymatgen.analysis.local_env import VoronoiNN
# Import the non-deprecated BrunnerNNReal class
try:
    # For newer versions of pymatgen
    from pymatgen.analysis.local_env import BrunnerNNReal
except ImportError:
    # Fallback for older versions
    from pymatgen.analysis.local_env import BrunnerNN_real as BrunnerNNReal
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import networkx as nx
from scipy.stats import entropy
import warnings
import concurrent.futures

# Filter out specific PyMatgen warnings
warnings.filterwarnings("ignore", message="CrystalNN: cannot locate an appropriate radius")
warnings.filterwarnings("ignore", message="No oxidation states specified on sites")
warnings.filterwarnings("ignore", message="BrunnerNN_real is deprecated")

# Global caches to reduce repeated calculations
_element_data_cache = {}
_site_neighbors_cache = {}


@lru_cache(maxsize=1000)
def get_element_symbol(atom_label):
    """Optimized with caching for frequent calls."""
    symbol = ''.join(c for c in atom_label if c.isalpha())
    if len(symbol) > 1:
        symbol = symbol[0].upper() + symbol[1].lower()
    else:
        symbol = symbol.upper()
    try:
        Element(symbol)
        return symbol
    except ValueError:
        return None


@lru_cache(maxsize=1000)
def get_element(symbol):
    """Cache Element objects to avoid repeated initialization."""
    try:
        return Element(symbol)
    except ValueError:
        return None


def get_element_data(element_symbol):
    """Get cached element data or compute it once."""
    if element_symbol in _element_data_cache:
        return _element_data_cache[element_symbol]
    
    element = get_element(element_symbol)
    if not element:
        return None
    
    data = {
        'atomic_mass': element.atomic_mass,
        'atomic_radius': element.atomic_radius,
        'van_der_waals_radius': element.van_der_waals_radius,
        'ionization_energy': element.ionization_energy,
        'electron_affinity': element.electron_affinity,
        'X': element.X,
        'is_metal': element.is_metal,
        'period': getattr(element, 'period', None),
        'group': getattr(element, 'group', None),
        'block': getattr(element, 'block', None),
        'oxidation_states': getattr(element, 'oxidation_states', []),
        'common_oxidation_states': getattr(element, 'common_oxidation_states', [])
    }
    
    _element_data_cache[element_symbol] = data
    return data


def calculate_molecular_mass(atom_labels):
    """Optimized by using cached element data."""
    return sum(get_element_data(el)['atomic_mass'] for el in map(get_element_symbol, atom_labels) if el and get_element_data(el))


def calculate_density_and_void_volume(structure, atom_labels):
    """Optimized by direct volume access and cached elements."""
    try:
        volume = structure.volume
        total_mass = calculate_molecular_mass(atom_labels)
        density = total_mass / volume if volume > 0 and total_mass > 0 else None
        
        # Calculate atomic volume more precisely
        atomic_volumes = []
        for label in atom_labels:
            symbol = get_element_symbol(label)
            if symbol:
                data = get_element_data(symbol)
                if data and data['atomic_radius']:
                    atomic_volumes.append((4/3) * math.pi * (data['atomic_radius'] ** 3))
        
        total_atomic_volume = sum(atomic_volumes)
        void_volume = max(volume - total_atomic_volume, 0) if volume else None

        return {
            'density': round(density, 4) if density else None,
            'void_volume': round(void_volume, 4) if void_volume else None,
            'void_fraction': round(void_volume/volume, 4) if void_volume and volume else None
        }
    except Exception as e:
        print(f"Error in density calculation: {e}")
        return {
            'density': None,
            'void_volume': None,
            'void_fraction': None
        }


def calculate_crystal_symmetry(structure):
    """Optimized by reusing SpacegroupAnalyzer."""
    try:
        analyzer = SpacegroupAnalyzer(structure)
        return {
            'spacegroup': analyzer.get_space_group_symbol(),
            'crystal_system': analyzer.get_crystal_system(),
            'point_group': analyzer.get_point_group_symbol()
        }
    except Exception as e:
        print(f"Error in symmetry analysis: {e}")
        return {
            'spacegroup': None,
            'crystal_system': None,
            'point_group': None
        }


def detect_rings_optimized(structure, max_ring_size=6):
    """
    Optimized ring detection using:
    - VoronoiNN for faster neighbor finding
    - Limited ring size search
    - Early termination for large structures
    """
    try:
        # Skip for large structures
        if len(structure) > 500:
            return {
                'num_rings': 0,
                'ring_sizes': [],
                'num_3_membered_rings': 0,
                'num_4_membered_rings': 0,
                'num_5_membered_rings': 0,
                'num_6_membered_rings': 0
            }

        # Use structure id for caching
        structure_id = id(structure)
        
        # Use VoronoiNN which is faster than other neighbor finders
        vnn = VoronoiNN(tol=0.5)  # Increased tolerance for faster computation
        G = nx.Graph()

        # Build graph with neighbor information
        for i in range(len(structure)):
            neighbors = vnn.get_nn_info(structure, i)
            for neighbor in neighbors:
                j = neighbor['site_index']
                if i < j:  # Avoid duplicate edges
                    G.add_edge(i, j)

        # Find rings of size 3-6 only
        ring_sizes = []
        # Use a more efficient algorithm for small rings
        for n in range(3, min(max_ring_size + 1, 7)):
            cycles = nx.simple_cycles(G.subgraph(list(range(min(100, len(structure))))))
            for cycle in cycles:
                if len(cycle) == n:
                    ring_sizes.append(n)
                    # Limit to max 100 rings per size for performance
                    if ring_sizes.count(n) >= 100:
                        break

        return {
            'num_rings': len(ring_sizes),
            'ring_sizes': ring_sizes,
            'num_3_membered_rings': ring_sizes.count(3),
            'num_4_membered_rings': ring_sizes.count(4),
            'num_5_membered_rings': ring_sizes.count(5),
            'num_6_membered_rings': ring_sizes.count(6)
        }
    except Exception as e:
        print(f"Optimized ring detection error: {e}")
        return {
            'num_rings': 0,
            'ring_sizes': [],
            'num_3_membered_rings': 0,
            'num_4_membered_rings': 0,
            'num_5_membered_rings': 0,
            'num_6_membered_rings': 0
        }


def get_neighbors(structure, i, nn=None):
    """Cache neighbor calculations to avoid repeated calls."""
    structure_id = id(structure)
    
    if structure_id not in _site_neighbors_cache:
        _site_neighbors_cache[structure_id] = {}
    
    if i in _site_neighbors_cache[structure_id]:
        return _site_neighbors_cache[structure_id][i]
    
    if nn is None:
        nn = BrunnerNNReal()
    
    neighbors = nn.get_nn_info(structure, i)
    _site_neighbors_cache[structure_id][i] = neighbors
    return neighbors


def calculate_hybridization_optimized(structure, neighbor_cache=None):
    """
    Faster hybridization calculation with better caching.
    """
    sp3 = sp2 = sp = 0
    
    try:
        nn = BrunnerNNReal()
        for i in range(len(structure)):
            if neighbor_cache and i in neighbor_cache:
                cn = neighbor_cache[i]
            else:
                neighbors = get_neighbors(structure, i, nn)
                cn = len(neighbors)
                if neighbor_cache is not None:
                    neighbor_cache[i] = cn

            if cn == 4: sp3 += 1
            elif cn == 3: sp2 += 1
            elif cn == 2: sp += 1

        return {
            'num_sp3_atoms': sp3,
            'num_sp2_atoms': sp2,
            'num_sp_atoms': sp
        }
    except Exception as e:
        print(f"Optimized hybridization error: {e}")
        return {
            'num_sp3_atoms': 0,
            'num_sp2_atoms': 0,
            'num_sp_atoms': 0
        }


def calculate_element_percentages(atom_labels):
    """Calculate the percentage of various elements in the structure."""
    percentages = {
        'carbon_percentage': 0.0,
        'nitrogen_percentage': 0.0,
        'oxygen_percentage': 0.0,
        'hydrogen_percentage': 0.0,
        'metal_percentage': 0.0,
        'metal_elements': []
    }
    
    try:
        total_atoms = len(atom_labels)
        if total_atoms == 0:
            return percentages
            
        element_counts = {
            'C': 0, 'N': 0, 'O': 0, 'H': 0,
            'metal': 0, 'metal_list': set()
        }
        
        for label in atom_labels:
            element = get_element_symbol(label)
            if not element:
                continue
                
            if element in ['C', 'N', 'O', 'H']:
                element_counts[element] += 1
                
            data = get_element_data(element)
            if data and data['is_metal']:
                element_counts['metal'] += 1
                element_counts['metal_list'].add(element)
                
        percentages = {
            'carbon_percentage': round((element_counts['C'] / total_atoms) * 100, 3),
            'nitrogen_percentage': round((element_counts['N'] / total_atoms) * 100, 3),
            'oxygen_percentage': round((element_counts['O'] / total_atoms) * 100, 3),
            'hydrogen_percentage': round((element_counts['H'] / total_atoms) * 100, 3),
            'metal_percentage': round((element_counts['metal'] / total_atoms) * 100, 3),
            'metal_elements': list(element_counts['metal_list'])
        }
        
    except Exception as e:
        print(f"Error calculating element percentages: {e}")
    
    return percentages


def calculate_element_diversity(atom_labels):
    """
    Calculate diversity metrics for elements in the structure.
    """
    try:
        # Extract element symbols
        elements = [get_element_symbol(label) for label in atom_labels if get_element_symbol(label)]
        
        # Count distinct elements
        distinct_elements = set(elements)
        num_distinct_elements = len(distinct_elements)
        
        # Calculate element frequency
        element_counts = Counter(elements)
        
        # Calculate Shannon entropy as a measure of diversity
        frequencies = [count/len(elements) for count in element_counts.values()]
        shannon_entropy = entropy(frequencies, base=2)
        
        # Maximum possible entropy for this number of elements
        max_entropy = math.log2(num_distinct_elements) if num_distinct_elements > 0 else 0
        
        # Normalized entropy (0-1 scale)
        normalized_entropy = shannon_entropy / max_entropy if max_entropy > 0 else 0
        
        return {
            'num_distinct_elements': num_distinct_elements,
            'element_diversity_entropy': round(shannon_entropy, 3),
            'element_diversity_normalized': round(normalized_entropy, 3)
        }
    except Exception as e:
        print(f"Error calculating element diversity: {e}")
        return {
            'num_distinct_elements': 0,
            'element_diversity_entropy': None,
            'element_diversity_normalized': None
        }


def calculate_periodic_table_stats(atom_labels):
    """
    Calculate statistics about element distribution across periodic table.
    """
    try:
        elements = [get_element_symbol(label) for label in atom_labels if get_element_symbol(label)]
        
        if not elements:
            return {}
            
        # Count elements by periods and groups
        periods = []
        groups = []
        block_counts = {'s': 0, 'p': 0, 'd': 0, 'f': 0}
        
        for el in elements:
            element_data = get_element_data(el)
            if not element_data:
                continue
                
            if element_data['period']:
                periods.append(element_data['period'])
                
            if element_data['group']:
                groups.append(element_data['group'])
                
            if element_data['block'] in block_counts:
                block_counts[element_data['block']] += 1
        
        # Calculate statistics
        result = {}
        
        if periods:
            avg_period = sum(periods) / len(periods)
            result['average_period'] = round(avg_period, 3)
            
        if groups:
            avg_group = sum(groups) / len(groups)
            result['average_group'] = round(avg_group, 3)
        
        # Calculate block percentages
        block_total = sum(block_counts.values())
        if block_total > 0:
            result['s_block_percentage'] = round((block_counts['s'] / block_total) * 100, 3)
            result['p_block_percentage'] = round((block_counts['p'] / block_total) * 100, 3)
            result['d_block_percentage'] = round((block_counts['d'] / block_total) * 100, 3)
            result['f_block_percentage'] = round((block_counts['f'] / block_total) * 100, 3)
        
        return result
    except Exception as e:
        print(f"Error calculating periodic table stats: {e}")
        return {}


def calculate_elemental_properties(atom_labels):
    """
    Calculate average values of various elemental properties.
    """
    try:
        elements = [get_element_symbol(label) for label in atom_labels if get_element_symbol(label)]
        
        if not elements:
            return {}
            
        # Initialize property lists
        atomic_radii = []
        van_der_waals_radii = []
        first_ionization_energies = []
        electron_affinities = []
        electronegativities = []
        atomic_masses = []
        
        # Collect properties for each element
        for el in elements:
            element_data = get_element_data(el)
            if not element_data:
                continue
                
            if element_data['atomic_radius']:
                atomic_radii.append(element_data['atomic_radius'])
                
            if element_data['van_der_waals_radius']:
                van_der_waals_radii.append(element_data['van_der_waals_radius'])
                
            if element_data['ionization_energy']:
                first_ionization_energies.append(element_data['ionization_energy'])
                
            if element_data['electron_affinity']:
                electron_affinities.append(element_data['electron_affinity'])
                
            if element_data['X']:
                electronegativities.append(element_data['X'])
                
            if element_data['atomic_mass']:
                atomic_masses.append(element_data['atomic_mass'])
        
        # Calculate averages
        result = {}
        
        if atomic_radii:
            result['average_atomic_radius'] = round(sum(atomic_radii) / len(atomic_radii), 3)
            result['min_atomic_radius'] = round(min(atomic_radii), 3)
            result['max_atomic_radius'] = round(max(atomic_radii), 3)
            
        if van_der_waals_radii:
            result['average_van_der_waals_radius'] = round(sum(van_der_waals_radii) / len(van_der_waals_radii), 3)
            
        if first_ionization_energies:
            result['average_ionization_energy'] = round(sum(first_ionization_energies) / len(first_ionization_energies), 3)
            
        if electron_affinities:
            result['average_electron_affinity'] = round(sum(electron_affinities) / len(electron_affinities), 3)
            
        if electronegativities:
            result['average_electronegativity'] = round(sum(electronegativities) / len(electronegativities), 3)
            result['electronegativity_difference'] = round(max(electronegativities) - min(electronegativities), 3)
            
        if atomic_masses:
            result['average_atomic_mass'] = round(sum(atomic_masses) / len(atomic_masses), 3)
        
        return result
    except Exception as e:
        print(f"Error calculating elemental properties: {e}")
        return {}


def add_oxidation_states(structure):
    """
    Add estimated oxidation states to a structure more efficiently.
    """
    try:
        # Create a copy of the structure to avoid modifying the original
        structure_with_oxi = structure.copy()
        
        # Get common oxidation states for each element
        for i, site in enumerate(structure_with_oxi):
            try:
                element_symbol = str(site.specie.symbol)
                
                # Skip if the element symbol is invalid
                if not element_symbol or element_symbol == "-1":
                    continue
                    
                element_data = get_element_data(element_symbol)
                if not element_data:
                    continue
                    
                oxi_states = element_data['common_oxidation_states']
                
                # If the element has common oxidation states, use the first one
                if oxi_states:
                    # Replace the site with one that has an oxidation state
                    structure_with_oxi[i] = site.specie.oxidation_state = oxi_states[0]
            except Exception:
                # Skip problematic sites instead of failing completely
                continue
        
        return structure_with_oxi
    except Exception as e:
        print(f"Warning: Could not add oxidation states: {e}")
        return structure


def calculate_oxidation_states(atom_labels):
    """
    Calculate oxidation state statistics more efficiently.
    """
    try:
        elements = [get_element_symbol(label) for label in atom_labels if get_element_symbol(label)]
        
        if not elements:
            return {}
            
        # Get all possible oxidation states
        all_states = []
        common_states = []
        
        for el in elements:
            element_data = get_element_data(el)
            if not element_data:
                continue
                
            if element_data['oxidation_states']:
                all_states.extend(element_data['oxidation_states'])
                
            if element_data['common_oxidation_states']:
                common_states.extend(element_data['common_oxidation_states'])
        
        # Calculate statistics
        result = {}
        
        if all_states:
            result['min_oxidation_state'] = min(all_states)
            result['max_oxidation_state'] = max(all_states)
            result['average_oxidation_state'] = round(sum(common_states) / len(common_states), 3) if common_states else None
            result['oxidation_state_range'] = max(all_states) - min(all_states)
        
        return result
    except Exception as e:
        print(f"Error calculating oxidation states: {e}")
        return {}


def calculate_bond_valence_sums(structure):
    """
    Estimate bond valence sums for the structure with optimized performance.
    """
    try:
        # Add oxidation states to the structure to avoid warnings
        structure_with_oxi = add_oxidation_states(structure)
        
        # Skip for large structures - this is computationally expensive
        if len(structure) > 500:
            return {}
        
        # Bond valence parameters (simplified)
        # R0 parameters for common bonds
        bond_params = {
            ('Si', 'O'): 1.624,
            ('Al', 'O'): 1.651,
            ('Ca', 'O'): 2.14,
            ('Na', 'O'): 2.28,
            ('K', 'O'): 2.42,
            ('Mg', 'O'): 2.08,
            ('Fe', 'O'): 1.78,
            ('Ti', 'O'): 1.815,
            ('P', 'O'): 1.617,
            ('Zn', 'O'): 1.704,
        }
        
        # Constant for bond valence calculation
        b = 0.37  # Empirical constant
        
        # Calculate bond valences
        # Use BrunnerNNReal instead of BrunnerNN_real (deprecated)
        nn = BrunnerNNReal()
        valences = []
        
        # Process a subset of sites for very large structures
        max_sites = min(len(structure_with_oxi), 200)  # limit to max 200 sites
        
        for i in range(max_sites):
            site = structure_with_oxi[i]
            element_i = str(site.specie.symbol)
            valence_sum = 0
            
            neighbors = get_neighbors(structure_with_oxi, i, nn)
            for neighbor in neighbors:
                j = neighbor['site_index']
                site_j = structure_with_oxi[j]
                element_j = str(site_j.specie.symbol)
                
                # Calculate distance directly from the two sites
                distance = site.distance(site_j)
                
                # Look up bond parameter or skip
                bond_key = (element_i, element_j)
                if bond_key in bond_params:
                    r0 = bond_params[bond_key]
                    valence = math.exp((r0 - distance) / b)
                    valence_sum += valence
                else:
                    # Try reverse order
                    bond_key = (element_j, element_i)
                    if bond_key in bond_params:
                        r0 = bond_params[bond_key]
                        valence = math.exp((r0 - distance) / b)
                        valence_sum += valence
            
            if valence_sum > 0:
                valences.append(valence_sum)
        
        # Calculate statistics
        if valences:
            avg_valence = sum(valences) / len(valences)
            min_valence = min(valences)
            max_valence = max(valences)
            valence_std = np.std(valences)
            
            return {
                'average_bond_valence_sum': round(avg_valence, 3),
                'min_bond_valence_sum': round(min_valence, 3),
                'max_bond_valence_sum': round(max_valence, 3),
                'bond_valence_sum_std': round(valence_std, 3)
            }
        
        return {}
    except Exception as e:
        print(f"Error calculating bond valence sums: {e}")
        return {}


def calculate_element_ratios(atom_labels):
    """
    Calculate important element ratios for materials science.
    """
    try:
        # Count elements
        element_counts = {}
        for label in atom_labels:
            element = get_element_symbol(label)
            if element:
                element_counts[element] = element_counts.get(element, 0) + 1
        
        # Initialize ratios
        ratios = {}
        
        # Si/Al ratio (important for zeolites)
        if 'Al' in element_counts and element_counts['Al'] > 0:
            si_al_ratio = element_counts.get('Si', 0) / element_counts['Al']
            ratios['Si_Al_ratio'] = round(si_al_ratio, 3)
        
        # Metal/O ratio (important for oxides)
        total_metals = sum(element_counts.get(el, 0) for el in element_counts 
                          if el and get_element_data(el) and get_element_data(el)['is_metal'])
        oxygen_count = element_counts.get('O', 0)
        
        if oxygen_count > 0:
            metal_o_ratio = total_metals / oxygen_count
            ratios['metal_O_ratio'] = round(metal_o_ratio, 3)
        
        # C/H ratio (important for organic materials)
        carbon_count = element_counts.get('C', 0)
        hydrogen_count = element_counts.get('H', 0)
        
        if hydrogen_count > 0:
            c_h_ratio = carbon_count / hydrogen_count
            ratios['C_H_ratio'] = round(c_h_ratio, 3)
        
        # N/C ratio (important for nitrogen-doped materials)
        nitrogen_count = element_counts.get('N', 0)
        
        if carbon_count > 0:
            n_c_ratio = nitrogen_count / carbon_count
            ratios['N_C_ratio'] = round(n_c_ratio, 3)
        
        return ratios
    except Exception as e:
        print(f"Error calculating element ratios: {e}")
        return {}