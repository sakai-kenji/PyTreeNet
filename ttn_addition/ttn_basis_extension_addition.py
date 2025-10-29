"""
TTN Basis Extension Addition Algorithm Outline

This file provides a comprehensive outline for generalizing the MPS basis extension 
algorithm to work with arbitrary tree tensor networks (TTNs).

Based on the MPS implementation in mps_basis_extension_addition.py, this outlines
the key generalizations needed for tree structures.
"""

import numpy as np
import copy
from pytreenet.core.node import Node
from pytreenet.core.ttn import TreeTensorNetwork
from pytreenet.ttns.ttns import TreeTensorNetworkState
from pytreenet.util.tensor_splitting import tensor_svd, truncated_tensor_svd, SVDParameters
from pytreenet.contractions.tree_contraction import completely_contract_tree
from ttn_addition import add_ttns_direct_form
from mps_particle_hopping import _mps_particle_hopping

def ttn_basis_extension_addition(ttn1, ttn2, svd_params=None):
    """
    Main algorithm structure for TTN basis extension addition.

    Args:
        ttn1, ttn2: TreeTensorNetworkState - The two TTNS to add
        svd_params: SVDParameters - Optional SVD truncation parameters

    Returns:
        TreeTensorNetworkState: Sum |ψ₁⟩ + |ψ₂⟩

    Raises:
        TypeError: If inputs are not TreeTensorNetworkState instances
    """
    
    # linearization order
    linearization_order = ttn1.linearise()
    
    initial_center = linearization_order[0]
    ttn1.canonical_form(orthogonality_center_id=initial_center)
    ttn2.canonical_form(orthogonality_center_id=initial_center)
    
    for i, current_node in enumerate(linearization_order[:-1]):  # Skip last node
        # print(f"Processing node {current_node} (step {i+1}/{len(linearization_order)-1})")
        next_center = linearization_order[i + 1]

        _apply_basis_extension_to_node(ttn1, ttn2, current_node, svd_params)

        current_node_obj = ttn1.nodes[current_node]
        are_neighbors = current_node_obj.parent == next_center

        if not are_neighbors:
            # print(f"Branch change: moving orthogonality center to {next_center}")
            ttn1.move_orthogonalization_center(next_center)
            ttn2.move_orthogonalization_center(next_center)
        else:
            # print(f"Same branch: updating orthogonality center to {current_node}")
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

    # Step B: SVD with sign stabilization (same as MPS)
    V, D, Vt = _stable_svd_decomposition(combined_proj, svd_params)

    # Step C: Contract-extend-split (you handle tensor splitting)
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
        projector: The projector tensor to decompose
        svd_params: SVDParameters - Optional truncation parameters

    Same sign stabilization as MPS version to ensure numerical stability.
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

    # Apply sign stabilization (same as MPS)
    for i in range(min(U.shape[-1], Vt.shape[0])):
        u_vec = U.reshape(-1, U.shape[-1])[:, i]
        first_nonzero = np.argmax(np.abs(u_vec))

        if u_vec[first_nonzero] < 0:
            U.reshape(-1, U.shape[-1])[:, i] *= -1
            Vt.reshape(Vt.shape[0], -1)[i, :] *= -1

    return U, S, Vt


def _contract_and_extend_with_parent(ttn1, ttn2, node_id, parent_id, V, Vt):
    # print(f"\n=== Processing node {node_id} with parent {parent_id} ===")

    # Step 2: Get leg specifications
    legs_a, legs_c = ttn1.legs_before_combination(parent_id, node_id)

    node_a1 = ttn1.nodes[parent_id]
    node_c1 = ttn1.nodes[node_id]

    # Save original position of the node in parent's children list
    original_node_position_1 = ttn1.nodes[parent_id].children.index(node_id)
    original_node_position_2 = ttn2.nodes[parent_id].children.index(node_id)

    # Step 3: Contract nodes
    temp_contracted_id = f"temp_{parent_id}_{node_id}"
    ttn1.contract_nodes(parent_id, node_id, new_identifier=temp_contracted_id)
    contracted_tensor1 = ttn1.tensors[temp_contracted_id]
    ttn2.contract_nodes(parent_id, node_id, new_identifier=temp_contracted_id)
    contracted_tensor2 = ttn2.tensors[temp_contracted_id]
    
    # Step 4: Apply basis extension 
    extended_tensor1, extended_tensor2, V_final = _apply_basis_extension_transform(
        contracted_tensor1, contracted_tensor2, V, Vt, ttn1.nodes[temp_contracted_id], node_a1, node_c1

    )
    
    # Step 5: Split
    ttn1.split_node_replace(
        temp_contracted_id,
        extended_tensor1,  # New parent tensor
        V_final,          # New node tensor
        parent_id,        # Parent identifier
        node_id,          # Node identifier
        legs_a,           # Parent leg specification
        legs_c            # Node leg specification
    )

    ttn2.split_node_replace(
        temp_contracted_id,
        extended_tensor2,  # New parent tensor
        V_final,          # New node tensor
        parent_id,        # Parent identifier
        node_id,          # Node identifier
        legs_a,           # Parent leg specification
        legs_c            # Node leg specification
    )

    # print(f"=== End processing node {node_id} ===\n")


def _apply_basis_extension_transform(tensor1, tensor2, V, Vt, contracted_node,  node_a1, node_c1):

    if node_a1.is_root():
        # Create indices [0, 1, ..., num_open_legs-1] then offset them
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

def test_ttn_basis_extension():
    """
    Test framework for TTN basis extension algorithm.
    
    Test against different TTN topologies to ensure generalization works.
    """
    
    # Test with different TTN structures
    test_cases = [
        ("small_ttn", create_small_test_ttn()),
        ("binary_tree", create_binary_tree_ttn(depth=2)),
        ("star_topology", create_star_ttn(num_leaves=4)),
    ]
    
    for name, (ttn1, ttn2) in test_cases:
        print(f"\nTesting {name}...")
        
        # Direct addition (reference)
        ttn1_copy = copy.deepcopy(ttn1)
        ttn2_copy = copy.deepcopy(ttn2)
        direct_result = add_ttns_direct_form(ttn1_copy, ttn2_copy)
        direct_contracted = completely_contract_tree(direct_result)
        
        # Basis extension addition
        basis_result = ttn_basis_extension_addition(ttn1, ttn2)
        basis_contracted = completely_contract_tree(basis_result)
        
        # Compare results
        if isinstance(direct_contracted, tuple):
            direct_contracted = direct_contracted[0]
        if isinstance(basis_contracted, tuple):
            basis_contracted = basis_contracted[0]
            
        match = np.allclose(direct_contracted, basis_contracted, rtol=1e-8)
        print(f"{name} results match: {match}")


def create_star_test_ttn():
    """Create star TTN with 1 root and 4 children for testing."""
    from pytreenet.ttns.ttns import TreeTensorNetworkState
    from pytreenet.core.node import Node
    import numpy as np
    
    # Create two identical star TTNs for testing
    ttn1 = TreeTensorNetworkState()
    ttn2 = TreeTensorNetworkState()
    
    # Add root node (connects 4 children + has physical leg)
    root_tensor1 = np.random.rand(2, 2, 2, 2, 3)  # [child1, child2, child3, child4, physical]
    root_tensor2 = root_tensor1 * 0.5
    root_node1 = Node(identifier="root")
    root_node2 = Node(identifier="root")
    ttn1.add_root(root_node1, root_tensor1)
    ttn2.add_root(root_node2, root_tensor2)
    
    # Add 4 children
    for i in range(4):
        child_tensor1 = np.random.rand(2, 2)  # [parent, physical]
        child_tensor2 = child_tensor1 * 0.7
        child_node1 = Node(identifier=f"child_{i}")
        child_node2 = Node(identifier=f"child_{i}")
        ttn1.add_child_to_parent(child_node1, child_tensor1, 0, "root", i)
        ttn2.add_child_to_parent(child_node2, child_tensor2, 0, "root", i)
    
    return ttn1, ttn2


def test_depth_n_tree():
    """Test TTN basis extension with depth 5 trees from generate_tree_pair."""
    print("Testing depth n TTN basis extension...")
    
    # Import the tree generation function
    from ttn_addition_benchmark import generate_tree_pair
    
    # Create test TTNs with depth
    # ttn1, ttn2 = generate_tree_pair(depth=5, leaf_probability=0, branching_factor=1)
    ttn1, ttn2 = generate_tree_pair(depth=3, leaf_probability=0, branching_factor=2, physical_dim=2)
    
    print(f"Generated trees with {len(ttn1.nodes)} nodes each")
    
    # Create copies for direct addition
    ttn1_copy = copy.deepcopy(ttn1)
    ttn2_copy = copy.deepcopy(ttn2)
    
    # Direct addition (reference)
    print("Computing direct addition...")
    direct_result = add_ttns_direct_form(ttn1_copy, ttn2_copy)
    direct_contracted = completely_contract_tree(direct_result)
    print(direct_contracted)
    
    # Basis extension addition
    max_bond = 6
    svd_params = SVDParameters(
                    max_bond_dim=max_bond,
                    renorm=True,
                    rel_tol=float("-inf"),
                    total_tol=float("-inf")
                )
    # svd_params = None
    print("Computing basis extension addition...")
    basis_result = ttn_basis_extension_addition(ttn1, ttn2, svd_params)
    basis_contracted = completely_contract_tree(basis_result)
    
    print(f"Direct result ordering: {direct_contracted[1] if isinstance(direct_contracted, tuple) else 'No ordering'}")
    print(f"Basis result ordering: {basis_contracted[1] if isinstance(basis_contracted, tuple) else 'No ordering'}")
    
    # Handle tuple results and fix ordering if necessary
    direct_ordering = None
    basis_ordering = None
    
    if isinstance(direct_contracted, tuple):
        direct_ordering = direct_contracted[1]
        direct_contracted = direct_contracted[0]
    if isinstance(basis_contracted, tuple):
        basis_ordering = basis_contracted[1]
        basis_contracted = basis_contracted[0]
    
    print(basis_contracted)

    # If both have ordering information and they differ, transpose basis result to match direct ordering
    if direct_ordering is not None and basis_ordering is not None and direct_ordering != basis_ordering:
        print("Orderings differ, transposing basis result to match direct ordering...")
        # Create permutation to transform from basis_ordering to direct_ordering
        perm = [basis_ordering.index(site) for site in direct_ordering]
        basis_contracted = np.transpose(basis_contracted, perm)
        print(basis_contracted)
        print(f"Applied permutation: {perm}")
    
    # Compare results
    match = np.allclose(direct_contracted, basis_contracted, rtol=1e-8)
    print(f"Depth n tree results match: {match}")
    
    if match:
        print("✓ Depth n tree test passed!")
    else:
        print("✗ Depth n tree test failed!")
        print(f"Direct result shape: {direct_contracted.shape}")
        print(f"Basis extension result shape: {basis_contracted.shape}")
        if hasattr(direct_contracted, 'shape') and hasattr(basis_contracted, 'shape'):
            if direct_contracted.shape == basis_contracted.shape:
                print(f"Max difference: {np.max(np.abs(direct_contracted - basis_contracted))}")
    
    return match


def test_star_ttn():
    """Test star TTN with 1 root and 4 children."""
    print("Testing star TTN (1 root + 4 children)...")
    
    # Create star TTN
    ttn1, ttn2 = create_star_test_ttn()
    
    print(f"Generated star trees with {len(ttn1.nodes)} nodes each")
    print(f"Linearization order: {ttn1.linearise()}")
    
    # Create copies for direct addition
    ttn1_copy = copy.deepcopy(ttn1)
    ttn2_copy = copy.deepcopy(ttn2)
    
    # Direct addition (reference)
    print("Computing direct addition...")
    direct_result = add_ttns_direct_form(ttn1_copy, ttn2_copy)
    direct_contracted = completely_contract_tree(direct_result)
    print(direct_contracted)
    
    # Basis extension addition
    print("Computing basis extension addition...")
    basis_result = ttn_basis_extension_addition(ttn1, ttn2)
    basis_contracted = completely_contract_tree(basis_result)
    
    print(f"Direct result ordering: {direct_contracted[1] if isinstance(direct_contracted, tuple) else 'No ordering'}")
    print(f"Basis result ordering: {basis_contracted[1] if isinstance(basis_contracted, tuple) else 'No ordering'}")
    
    # Handle tuple results and fix ordering if necessary
    direct_ordering = None
    basis_ordering = None
    
    if isinstance(direct_contracted, tuple):
        direct_ordering = direct_contracted[1]
        direct_contracted = direct_contracted[0]
    if isinstance(basis_contracted, tuple):
        basis_ordering = basis_contracted[1]
        basis_contracted = basis_contracted[0]
    
    # If both have ordering information and they differ, transpose basis result to match direct ordering
    if direct_ordering is not None and basis_ordering is not None and direct_ordering != basis_ordering:
        print("Orderings differ, transposing basis result to match direct ordering...")
        # Create permutation to transform from basis_ordering to direct_ordering
        perm = [basis_ordering.index(site) for site in direct_ordering]
        basis_contracted = np.transpose(basis_contracted, perm)
        print(basis_contracted)
        print(f"Applied permutation: {perm}")
    
    # Compare results
    match = np.allclose(direct_contracted, basis_contracted, rtol=1e-8)
    print(f"Star TTN results match: {match}")
    
    if not match:
        print(f"Direct result shape: {direct_contracted.shape}")
        print(f"Basis extension result shape: {basis_contracted.shape}")
        print(f"Max difference: {np.max(np.abs(direct_contracted - basis_contracted))}")
        
        # Show detailed comparison for first few elements
        print("\\nFirst few elements comparison:")
        flat_direct = direct_contracted.flatten()
        flat_basis = basis_contracted.flatten()
        for i in range(min(8, len(flat_direct))):
            print(f"[{i}] Direct: {flat_direct[i]:.6f}, Basis: {flat_basis[i]:.6f}, Diff: {abs(flat_direct[i] - flat_basis[i]):.6f}")
    
    return match

if __name__ == "__main__":
    test_depth_n_tree()


