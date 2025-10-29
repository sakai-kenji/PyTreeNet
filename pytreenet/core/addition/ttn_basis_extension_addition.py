import numpy as np
import copy
from pytreenet.core.node import Node
from pytreenet.core.ttn import TreeTensorNetwork
from pytreenet.ttns.ttns import TreeTensorNetworkState
from pytreenet.util.tensor_splitting import tensor_svd, truncated_tensor_svd, SVDParameters
from pytreenet.contractions.tree_contraction import completely_contract_tree

def ttn_basis_extension_addition(ttn1, ttn2, svd_params=None):
    """
    TTN basis extension addition, adapted from Time-dependent variational principle with ancillary Krylov subspace by Yang and White (2020)

    Args:
        ttn1, ttn2: TreeTensorNetworkState - The two TTNS to add
        svd_params: SVDParameters - Optional SVD truncation parameters

    Returns:
        TreeTensorNetworkState: Sum |ψ₁⟩ + |ψ₂⟩

    Raises:
        TypeError: If inputs are not TreeTensorNetworkState instances
    """
    
    linearization_order = ttn1.linearise()    
    initial_center = linearization_order[0]

    ttn1.canonical_form(orthogonality_center_id=initial_center)
    ttn2.canonical_form(orthogonality_center_id=initial_center)
    
    for i, current_node in enumerate(linearization_order[:-1]):  # Skip last node (Root)
        next_center = linearization_order[i + 1]

        _apply_basis_extension_to_node(ttn1, ttn2, current_node, svd_params)

        current_node_obj = ttn1.nodes[current_node]
        are_neighbors = current_node_obj.parent == next_center

        if not are_neighbors:
            ttn1.move_orthogonalization_center(next_center)
            ttn2.move_orthogonalization_center(next_center)
        else:
            ttn1.orthogonality_center_id = current_node
            ttn2.orthogonality_center_id = current_node
    
    # final orthogonality center, should be the root
    final_center = linearization_order[-1]
    center_tensor1 = ttn1.tensors[final_center]
    center_tensor2 = ttn2.tensors[final_center]
    combined_center = center_tensor1 + center_tensor2
    ttn1.replace_tensor(final_center, combined_center)
    
    return ttn1




def _apply_basis_extension_to_node(ttn1, ttn2, node_id, svd_params=None):

    node1 = ttn1.nodes[node_id]
    node2 = ttn2.nodes[node_id]
    tensor1 = ttn1.tensors[node_id]
    tensor2 = ttn2.tensors[node_id]

    parent_leg_tensor1 = ttn1.nodes[node_id].parent_leg
    parent_leg_tensor2 = ttn2.nodes[node_id].parent_leg

    # Step A: Compute projectors
    proj1 = _compute_projector(tensor1, parent_leg_tensor1)
    proj2 = _compute_projector(tensor2, parent_leg_tensor2)
    combined_proj = proj1 + proj2

    # Step B: SVD with sign stabilization
    V, D, Vt = _stable_svd_decomposition(combined_proj, svd_params)

    # Step C: Contract-extend-split
    parent_id = ttn1.nodes[node_id].parent
    _contract_and_extend_with_parent(ttn1, ttn2, node_id, parent_id, V, Vt)


def _compute_projector(tensor, parent_leg):
    """
    Compute projector (reduced density matrix) by contracting tensor with its conjugate, tracing out parent leg.
    """

    conj_tensor = np.conj(tensor)
    projector = np.tensordot(tensor, conj_tensor, ([parent_leg], [parent_leg]))
    return projector


def _stable_svd_decomposition(projector, svd_params=None):
    """
    SVD decomposition with deterministic sign convention and optional truncation.

    Args:
        projector
        svd_params (optional)
    """

    # Determine how to split tensor for SVD
    ndim = projector.ndim
    mid = ndim // 2

    u_legs = tuple(range(mid))
    v_legs = tuple(range(mid, ndim))

    # Perform SVD with optional truncation parameters
    if svd_params is not None:
        U, S, Vt = truncated_tensor_svd(projector, u_legs=u_legs, v_legs=v_legs,
                                        svd_params=svd_params)
    else:
        U, S, Vt = tensor_svd(projector, u_legs=u_legs, v_legs=v_legs)

    # Sort by singular values (descending)
    sort_idx = np.argsort(-S.real)
    S = S[sort_idx]
    U = U[..., sort_idx]
    Vt = Vt[sort_idx, ...]

    # Apply sign stabilization
    for i in range(min(U.shape[-1], Vt.shape[0])):
        u_vec = U.reshape(-1, U.shape[-1])[:, i]
        first_nonzero = np.argmax(np.abs(u_vec))

        if u_vec[first_nonzero] < 0:
            U.reshape(-1, U.shape[-1])[:, i] *= -1
            Vt.reshape(Vt.shape[0], -1)[i, :] *= -1

    return U, S, Vt


def _contract_and_extend_with_parent(ttn1, ttn2, node_id, parent_id, V, Vt):

    # Get leg specifications
    legs_a, legs_c = ttn1.legs_before_combination(parent_id, node_id)

    node_a1 = ttn1.nodes[parent_id]
    node_c1 = ttn1.nodes[node_id]

    # Save original position of the node in parent's children list
    original_node_position_1 = ttn1.nodes[parent_id].children.index(node_id)
    original_node_position_2 = ttn2.nodes[parent_id].children.index(node_id)

    # Contract nodes
    temp_contracted_id = f"temp_{parent_id}_{node_id}"

    ttn1.contract_nodes(parent_id, node_id, new_identifier=temp_contracted_id)
    contracted_tensor1 = ttn1.tensors[temp_contracted_id]

    ttn2.contract_nodes(parent_id, node_id, new_identifier=temp_contracted_id)
    contracted_tensor2 = ttn2.tensors[temp_contracted_id]
    
    # Apply basis extension 
    extended_tensor1, extended_tensor2, V_final = _apply_basis_extension_transform(
        contracted_tensor1, 
        contracted_tensor2, 
        V, 
        Vt, 
        ttn1.nodes[temp_contracted_id], 
        node_a1, 
        node_c1
    )
    
    # Split
    ttn1.split_node_replace(
        temp_contracted_id,
        extended_tensor1, # New parent tensor
        V_final,          # New node tensor
        parent_id,        # Parent identifier
        node_id,          # Node identifier
        legs_a,           # Parent leg specification
        legs_c            # Node leg specification
    )

    ttn2.split_node_replace(
        temp_contracted_id,
        extended_tensor2, # New parent tensor
        V_final,          # New node tensor
        parent_id,        # Parent identifier
        node_id,          # Node identifier
        legs_a,           # Parent leg specification
        legs_c            # Node leg specification
    )

def _apply_basis_extension_transform(tensor1, tensor2, V, Vt, contracted_node,  node_a1, node_c1):

    if node_a1.is_root():
        # Create indices [0, ..., num_open_legs-1] then offset them
        child_legs = list(leg + len(node_a1.children) - 2 for leg in node_c1.children_legs)
        # -1 becase a1 doesnt have a parent leg, -1 because when contracted the child leg between a1 and c1 doesn't exist anymore
        open_legs = list(leg + len(node_a1.open_legs) + len(contracted_node.children) for leg in range(len(node_c1.open_legs))) #because it doesnt have a parent
    else:
        child_legs = list(leg + len(node_a1.children) - 1 for leg in node_c1.children_legs)
        #-1 because when contracted the child leg between a1 and c1 doesn't exist anymore
        open_legs = list(leg + len(node_a1.open_legs) + len(contracted_node.children) + 1 for leg in range(len(node_c1.open_legs)))

    ac_contract_axis = child_legs + open_legs
    vt_contract_axis = list(range(1, Vt.ndim))
    V = np.transpose(V, (V.ndim - 1,) + tuple(range(V.ndim - 1)))

    # Apply the transformation
    extended_tensor1 = np.tensordot(tensor1, Vt, axes=(ac_contract_axis, vt_contract_axis))
    extended_tensor2 = np.tensordot(tensor2, Vt, axes=(ac_contract_axis, vt_contract_axis))
    
    return extended_tensor1, extended_tensor2, V
