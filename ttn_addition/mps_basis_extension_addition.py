import numpy as np
import copy
from pytreenet.core.node import Node
from pytreenet.core.ttn import TreeTensorNetwork
from pytreenet.util.tensor_splitting import tensor_svd
from pytreenet.contractions.node_contraction import contract_nodes
from pytreenet.contractions.tree_contraction import completely_contract_tree
from mps_particle_hopping import _mps_particle_hopping
from ttn_addition import add_ttns_direct_form

def basis_extension_addition(mps1, mps2):
    """
    Add two MPS using a modified basis extension algorithm as described in Yang, M., & White, S. R. (2020). Time-dependent variational principle 
    with ancillary Krylov subspace.

    Args:
        mps1: TreeTensorNetwork (MPS) - State |ψ⟩ 
        mps2: TreeTensorNetwork (MPS) - Reference state |φ⟩
        
    Returns:
        TreeTensorNetwork: Sum |ψ⟩ + |φ⟩ as an MPS
    """
    # Step a: Put both MPS in canonical form with orthogonal center at rightmost node
    rightmost_node = get_rightmost_node(mps1)
    mps1.canonical_form(orthogonality_center_id=rightmost_node)
    mps2.canonical_form(orthogonality_center_id=rightmost_node)
    
    # Process the nodes from right to left
    node_order = get_right_to_left_order(mps1)
    
    for j, node_id in enumerate(node_order):
        print("\n====================\n", node_id)
        if mps1.nodes[node_id].is_root():
            break
        # Get current node tensors
        tensor1 = mps1.tensors[node_id]
        tensor2 = mps2.tensors[node_id]
        parent_leg_tensor1 = mps1.nodes[node_id].parent_leg
        parent_leg_tensor2 = mps2.nodes[node_id].parent_leg
        
        proj1 = np.tensordot(tensor1, np.conj(tensor1), ([parent_leg_tensor1], [parent_leg_tensor1]))
        proj2 = np.tensordot(tensor2, np.conj(tensor2), ([parent_leg_tensor2], [parent_leg_tensor2]))
        combined_proj = proj1 + proj2
        
        legs = list(range(combined_proj.ndim))
        n = len(legs)
        mid = n // 2 
        legs_l = legs[mid:]

        V, D, Vt = tensor_svd(combined_proj, u_legs=legs[:mid], v_legs=legs_l)
        
        # Enforce deterministic sign convention: ensure the first nonzero element of each singular vector is positive
        for i in range(min(V.shape[-1], Vt.shape[0])):
            # For V, check the first non-zero element in the i-th column (flattened)
            v_col = V.reshape(-1, V.shape[-1])[:, i]
            first_nonzero_idx = np.argmax(np.abs(v_col))
            if v_col[first_nonzero_idx] < 0:
                # Flip both V column and Vt row
                V.reshape(-1, V.shape[-1])[:, i] *= -1
                Vt.reshape(Vt.shape[0], -1)[i, :] *= -1

        # Step d.1: Contract nodes i-1 and i, then split them back
        if j < len(node_order) - 1:  # Not the root
            prev_node_id = get_left_neighbor(mps1, node_id)
            if prev_node_id:
                # Get original nodes and tensors before any modifications
                node_a1 = mps1.nodes[prev_node_id]
                node_c1 = mps1.nodes[node_id]
                tensor_a1 = mps1.tensors[prev_node_id]

                node_a2 = mps2.nodes[prev_node_id]
                node_c2 = mps2.nodes[node_id]
                tensor_a2 = mps2.tensors[prev_node_id]
                
                # Get leg specifications before contracting
                legs_a, legs_c = mps1.legs_before_combination(prev_node_id, node_id)
                
                # Contract A with original C for MPS1
                temp_contracted_id = f"temp_{prev_node_id}_{node_id}"
                mps1.contract_nodes(prev_node_id, node_id, new_identifier=temp_contracted_id)
                ac_tensor1 = mps1.tensors[temp_contracted_id]
                
                # Contract A with original C for MPS2 (using external contract_nodes)
                mps2.contract_nodes(prev_node_id, node_id, new_identifier=temp_contracted_id)
                ac_tensor2 = mps2.tensors[temp_contracted_id]

                if not node_a1.is_root():
                    child_legs = list(leg + len(node_a1.children) - 2 for leg in node_c1.children_legs)
                    open_legs = list(leg + len(node_a1.open_legs) for leg in node_c1.open_legs) 
                else:
                    child_legs = list(leg + len(node_a1.children) - 1 for leg in node_c1.children_legs)
                    open_legs = list(leg + len(node_a1.open_legs) - 1 for leg in node_c1.open_legs)
                ac_contract_axis = child_legs + open_legs
                vt_contract_axis = list(range(1, Vt.ndim))
                V_new = np.transpose(V, (V.ndim - 1,) + tuple(range(V.ndim - 1)))

                child_legs = mps1.nodes[temp_contracted_id].children_legs
                if not node_a1.is_root():
                    open_legs = list(leg + len(node_a1.open_legs) for leg in node_c1.open_legs) 
                else:
                    open_legs = list(leg + len(node_a1.open_legs) - 1 for leg in node_c1.open_legs)
                ac_contract_axis = child_legs + open_legs
                vt_contract_axis = list(range(1, Vt.ndim))
                V = np.transpose(V, (V.ndim - 1,) + tuple(range(V.ndim - 1)))

                acv_tensor1 = np.tensordot(ac_tensor1, Vt, axes=(ac_contract_axis, vt_contract_axis))
                acv_tensor2 = np.tensordot(ac_tensor2, Vt, axes=(ac_contract_axis, vt_contract_axis))
                print(f"acv_tensor1 shape: {acv_tensor1.shape}")

                # Use split_node_replace to replace the contracted node with summed ACV† and V†
                mps1.split_node_replace(
                    temp_contracted_id,
                    acv_tensor1,  # tensor_a: ACV†₁ + ACV†₂
                    V,               # tensor_b: V†
                    prev_node_id,    # identifier_a
                    node_id,         # identifier_b  
                    legs_a,          # legs_a
                    legs_c           # legs_b
                )

                mps2.split_node_replace(
                    temp_contracted_id,
                    acv_tensor2,  # tensor_a: ACV†₁ + ACV†₂
                    V,               # tensor_b: V†
                    prev_node_id,    # identifier_a
                    node_id,         # identifier_b  
                    legs_a,          # legs_a
                    legs_c           # legs_b
                )

                print(mps1.tensors[prev_node_id].shape, mps1.tensors[node_id].shape)

    root_id = mps1.root_id
    new_root = mps1.tensors[root_id] + mps2.tensors[root_id]
    mps1.replace_tensor(root_id, new_root)

    return mps1


def get_rightmost_node(mps):
    """Get the rightmost (leaf) node ID."""
    for node_id, node in mps.nodes.items():
        if node.is_leaf() and not node.is_root():
            return node_id
    return mps.root_id  # fallback


def get_right_to_left_order(mps):
    """Get node processing order from right to left (leaf to root)."""
    order = []
    current = get_rightmost_node(mps)
    
    while current is not None:
        order.append(current)
        node = mps.nodes[current]
        if node.is_root():
            break
        current = node.parent
    
    return order


def get_left_neighbor(mps, node_id):
    """Get the left neighbor (parent) of a node in MPS."""
    node = mps.nodes[node_id]
    return node.parent if not node.is_root() else None


def test_basis_extension():
    """Test basis extension addition against direct addition."""
    print("Testing basis extension addition...")
    n_sites = 4
    # Create two identical MPS
    mps1 = _mps_particle_hopping(n_sites=n_sites)
    mps2 = _mps_particle_hopping(n_sites=n_sites)
    mps1_copy = copy.deepcopy(mps1)
    mps2_copy = copy.deepcopy(mps2)
    
    # Method 1: Direct addition (reference)
    print("Direct addition method...")
    # mps1_copy.canonical_form(orthogonality_center_id='site_3')
    # mps2_copy.canonical_form(orthogonality_center_id='site_3')
    direct_result = add_ttns_direct_form(mps1_copy, mps2_copy)
    direct_contracted = completely_contract_tree(direct_result)
    numeric_part = direct_contracted[0]
    rounded_numeric = np.round(numeric_part, 2)
    print(rounded_numeric)
    if isinstance(direct_contracted, tuple):
        direct_contracted = direct_contracted[0]
    
    # Method 2: Basis extension addition
    print("Basis extension method...")
    basis_result = basis_extension_addition(mps1, mps2)
    basis_contracted = completely_contract_tree(basis_result)
    numeric_part = basis_contracted[0]
    rounded_numeric = np.round(numeric_part, 2)
    print(rounded_numeric)    
    if isinstance(basis_contracted, tuple):
        basis_contracted = basis_contracted[0]
    
    # Compare results
    print(f"Direct result shape: {direct_contracted.shape}")
    print(f"Basis extension result shape: {basis_contracted.shape}")
    
    # Check if results are close
    print(f"Results match: {np.allclose(direct_contracted, basis_contracted, rtol=1e-8)}")
    
    return direct_contracted, basis_contracted


if __name__ == "__main__":
    test_basis_extension()

