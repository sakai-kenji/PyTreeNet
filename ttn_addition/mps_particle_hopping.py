import numpy as np
import copy
from pytreenet.core.node import Node
from pytreenet.core.ttn import TreeTensorNetwork
from pytreenet.contractions.tree_contraction import completely_contract_tree
from ttn_addition import add_ttns_direct_form


def _mps_particle_hopping(n_sites=4):
    """
    Create an MPS representing a particle hopping system on a 1D lattice.
    
    The state represents a superposition where a particle can be at any site:
    |ψ⟩ = (1/sqrt(n_sites))(|100...0⟩ + |010...0⟩ + |001...0⟩ + ... + |000...1⟩)
    
    Args:
        n_sites (int): Number of sites in the 1D lattice (default: 4)
    
    Returns:
        TreeTensorNetwork: MPS representing the particle hopping system
    """
    if n_sites < 2:
        raise ValueError("Need at least 2 sites for the MPS")
    
    ttn = TreeTensorNetwork()
    
    # Normalization factor for equal superposition of n_sites configurations
    psi = 1.0 / np.sqrt(n_sites) 
    ket0 = np.array([1, 0])
    ket1 = np.array([0, 1])

    tensor_0 = np.zeros((2, 2))
    tensor_0[0, :] = ket0
    tensor_0[1, :] = psi * ket1
    
    root_node = Node(identifier="site_0")
    ttn.add_root(root_node, tensor_0)
    
    # Intermediate sites (1 to n_sites-2): All have the same structure
    for i in range(1, n_sites - 1):
        tensor_i = np.zeros((2, 2, 2))
        tensor_i[0, 0, :] = ket0
        tensor_i[0, 1, :] = psi * ket1
        tensor_i[1, 1, :] = ket0
        
        node_i = Node(identifier=f"site_{i}")
        parent_id = f"site_{i-1}"
        parent_leg = 0 if i == 1 else 1
        ttn.add_child_to_parent(node_i, tensor_i, 0, parent_id, parent_leg)

    tensor_final = np.zeros((2, 2))
    tensor_final[0, :] = psi * ket1
    tensor_final[1, :] = ket0
    
    node_final = Node(identifier=f"site_{n_sites-1}")
    parent_id = f"site_{n_sites-2}"
    ttn.add_child_to_parent(node_final, tensor_final, 0, parent_id, 1)
    
    return ttn


if __name__ == "__main__":
    # Create the MPS
    mps = _mps_particle_hopping()
    mps1 = copy.deepcopy(mps)
    mps2 = copy.deepcopy(mps)
    mps1.canonical_form(orthogonality_center_id='site_0')
    mps2.canonical_form(orthogonality_center_id='site_0')

    doubled_mps = add_ttns_direct_form(mps1, mps2)
    doubled_result = completely_contract_tree(doubled_mps)
    if isinstance(doubled_result, tuple):
        doubled_result = doubled_result[0]  # Extract the tensor from tuple
    print(f"Doubled result shape: {doubled_result.shape}")
    print(f"Doubled result:\n{doubled_result}")
    print(mps1)