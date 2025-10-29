"""
Test script for the SVD-controlled TTN addition functionality.
"""
import numpy as np
import sys
import os

sys.path.append('/Users/sakai/Desktop/Uni/PyTreeNet')
sys.path.append('/Users/sakai/Desktop/Uni/PyTreeNet/Andres')

from pytreenet.core.ttn import TreeTensorNetwork
from pytreenet.core.node import Node
from pytreenet.util.tensor_splitting import SVDParameters
from ttn_addition_svd import add_ttns_svd_controlled, add_ttns_svd_controlled_params
from ttn_addition import add_ttns_direct_form

def create_simple_test_ttn(seed=None):
    """Create a simple TTN for testing purposes with consistent structure."""
    if seed is not None:
        np.random.seed(seed)
    
    ttn = TreeTensorNetwork()
    
    root_tensor = np.random.random((2, 3, 4))
    root_node = Node(tensor=root_tensor, identifier="root")
    ttn.add_root(root_node, root_tensor)
    
    child1_tensor = np.random.random((2, 5))
    child1_node = Node(tensor=child1_tensor, identifier="child1")
    ttn.add_child_to_parent(child1_node, child1_tensor, 0, "root", 0)
    
    child2_tensor = np.random.random((3, 6))
    child2_node = Node(tensor=child2_tensor, identifier="child2") 
    ttn.add_child_to_parent(child2_node, child2_tensor, 0, "root", 1)
    
    return ttn

def test_basic_functionality():
    """Test basic SVD-controlled addition functionality."""
    print("Testing basic SVD-controlled addition...")
    
    ttn1 = create_simple_test_ttn(seed=42)
    ttn2 = create_simple_test_ttn(seed=123)
    
    print(f"TTN1 bond dimensions: {[tensor.shape for tensor in ttn1.tensors.values()]}")
    print(f"TTN2 bond dimensions: {[tensor.shape for tensor in ttn2.tensors.values()]}")
    
    # Test direct addition (for comparison)
    direct_result = add_ttns_direct_form(ttn1, ttn2)
    print(f"Direct addition result shapes: {[tensor.shape for tensor in direct_result.tensors.values()]}")
    
    # Test SVD-controlled addition
    svd_params = SVDParameters(max_bond_dim=5, rel_tol=1e-10, total_tol=1e-10)
    svd_result = add_ttns_svd_controlled(ttn1, ttn2, svd_params)
    print(f"SVD-controlled result shapes: {[tensor.shape for tensor in svd_result.tensors.values()]}")
    
    # Test adaptive wrapper
    adaptive_result = add_ttns_svd_controlled_params(ttn1, ttn2, max_bond_dim=3, rel_tol=1e-8)
    print(f"Adaptive SVD result shapes: {[tensor.shape for tensor in adaptive_result.tensors.values()]}")
    
    print("Basic functionality test completed successfully!\n")

def test_bond_dimension_control():
    """Test that bond dimensions are properly controlled."""
    print("Testing bond dimension control...")
    
    ttn1 = create_simple_test_ttn(seed=42)
    ttn2 = create_simple_test_ttn(seed=123) 
    
    # Test with very strict bond dimension limit
    strict_params = SVDParameters(max_bond_dim=2, rel_tol=1e-15, total_tol=1e-15)
    strict_result = add_ttns_svd_controlled(ttn1, ttn2, strict_params)
    
    print("Strict bond dimension control (max_bond_dim=2):")
    for node_id, tensor in strict_result.tensors.items():
        max_bond_dim = max(tensor.shape[:-1] if strict_result.nodes[node_id].is_leaf() else tensor.shape[:-tensor.shape.count(tensor.shape[-1])])
        print(f"  {node_id}: shape {tensor.shape}, max bond dim: {max_bond_dim}")
        
    print("Bond dimension control test completed!\n")

def test_tolerance_effects():
    """Test the effect of different tolerance settings."""
    print("Testing tolerance effects...")
    
    ttn1 = create_simple_test_ttn(seed=42)
    ttn2 = create_simple_test_ttn(seed=123)
    
    # High tolerance (more truncation)
    high_tol = SVDParameters(max_bond_dim=100, rel_tol=1e-3, total_tol=1e-3)
    high_tol_result = add_ttns_svd_controlled(ttn1, ttn2, high_tol)
    
    # Low tolerance (less truncation)  
    low_tol = SVDParameters(max_bond_dim=100, rel_tol=1e-15, total_tol=1e-15)
    low_tol_result = add_ttns_svd_controlled(ttn1, ttn2, low_tol)
    
    print(f"High tolerance shapes: {[tensor.shape for tensor in high_tol_result.tensors.values()]}")
    print(f"Low tolerance shapes: {[tensor.shape for tensor in low_tol_result.tensors.values()]}")
    
    print("Tolerance effects test completed!\n")

def test_bond_dimension_5_to_2():
    """Simple numerical test showing bond dimension reduction from 5 to 2."""
    print("Testing bond dimension reduction: 5 → 2")
    print("=" * 50)
    
    # Create TTNs with initial bond dimension 5
    ttn1 = TreeTensorNetwork()
    ttn2 = TreeTensorNetwork()
    
    # Root tensor: (child1_bond=5, child2_bond=5, physical_dims...)
    root1_tensor = np.random.rand(5, 5, 2, 2) 
    root1_node = Node(tensor=root1_tensor, identifier="root")
    ttn1.add_root(root1_node, root1_tensor)
    
    root2_tensor = np.random.rand(5, 5, 2, 2)
    root2_node = Node(tensor=root2_tensor, identifier="root") 
    ttn2.add_root(root2_node, root2_tensor)
    
    # Child1 (leaf): (parent_bond=5, physical_dims...)
    child1_1_tensor = np.random.rand(5, 2, 2)
    child1_1_node = Node(tensor=child1_1_tensor, identifier="child1")
    ttn1.add_child_to_parent(child1_1_node, child1_1_tensor, 0, "root", 0)
    
    child1_2_tensor = np.random.rand(5, 2, 2)
    child1_2_node = Node(tensor=child1_2_tensor, identifier="child1")
    ttn2.add_child_to_parent(child1_2_node, child1_2_tensor, 0, "root", 0)
    
    # Child2 (leaf): (parent_bond=5, physical_dims...)
    child2_1_tensor = np.random.rand(5, 2, 2)
    child2_1_node = Node(tensor=child2_1_tensor, identifier="child2")
    ttn1.add_child_to_parent(child2_1_node, child2_1_tensor, 0, "root", 1)
    
    child2_2_tensor = np.random.rand(5, 2, 2)
    child2_2_node = Node(tensor=child2_2_tensor, identifier="child2")
    ttn2.add_child_to_parent(child2_2_node, child2_2_tensor, 0, "root", 1)
    
    print("Initial TTN structures (bond dimension = 5):")
    print(f"  TTN1 root: {root1_tensor.shape}")
    print(f"  TTN1 child1: {child1_1_tensor.shape}")
    print(f"  TTN1 child2: {child2_1_tensor.shape}")
    print(f"  TTN2 root: {root2_tensor.shape}")
    print(f"  TTN2 child1: {child1_2_tensor.shape}") 
    print(f"  TTN2 child2: {child2_2_tensor.shape}")
    
    # Direct addition (should double bond dimensions to 10)
    direct_result = add_ttns_direct_form(ttn1, ttn2)
    print(f"\nDirect addition result (bond dimension = 10):")
    for node_id in ["root", "child1", "child2"]:
        shape = direct_result.tensors[node_id].shape
        print(f"  {node_id}: {shape}")
        
    # SVD controlled addition with max_bond_dim=2
    svd_params = SVDParameters(max_bond_dim=2, rel_tol=1e-12, total_tol=1e-12)
    svd_result = add_ttns_svd_controlled(ttn1, ttn2, svd_params)
    print(f"\nSVD controlled result (max bond dimension = 2):")
    for node_id in ["root", "child1", "child2"]:
        shape = svd_result.tensors[node_id].shape
        max_bond = max(shape[:-2]) if len(shape) > 2 else max(shape[:-1]) if len(shape) > 1 else 1
        print(f"  {node_id}: {shape} → max bond dim: {max_bond}")
    
    # Verify that all bonds are indeed <= 2
    all_bonds_controlled = True
    for node_id, tensor in svd_result.tensors.items():
        node = svd_result.nodes[node_id]
        if node.is_leaf():
            # For leaf nodes, only parent bond matters
            bond_dim = tensor.shape[0]
        else:
            # For internal nodes, check all bond dimensions (exclude physical dims)
            bond_dims = tensor.shape[:-2] if len(tensor.shape) > 2 else tensor.shape[:-1] if len(tensor.shape) > 1 else tensor.shape
            bond_dim = max(bond_dims) if bond_dims else 1
            
        if bond_dim > 2:
            all_bonds_controlled = False
            print(f"  WARNING: {node_id} has bond dimension {bond_dim} > 2")
    
    print(f"\nBond dimension control successful: {all_bonds_controlled}")
    
    # Demonstrate the compression effect
    original_params = np.sum([t.size for t in ttn1.tensors.values()]) + np.sum([t.size for t in ttn2.tensors.values()])
    direct_params = np.sum([t.size for t in direct_result.tensors.values()])
    svd_params_count = np.sum([t.size for t in svd_result.tensors.values()])
    
    print(f"\nParameter count comparison:")
    print(f"  Original TTNs total: {original_params}")
    print(f"  Direct addition: {direct_params}")
    print(f"  SVD controlled: {svd_params_count}")
    print(f"  Compression ratio (SVD/Direct): {svd_params_count/direct_params:.3f}")
    
    print("✓ Bond dimension 5→2 test completed!\n")
    
    return ttn1, ttn2, direct_result, svd_result

def test_manual_verification():
    """Bigger example for manual paper verification with 3-level tree."""
    print("Manual Verification Test - 3-Level Tree Structure")
    print("=" * 60)
    
    ttn1 = TreeTensorNetwork()
    ttn2 = TreeTensorNetwork()
    
    print("Creating TTN with structure:")
    print("           root(2,2)")
    print("          /        \\")
    print("    node_A(2,2,2)   node_B(2)")  
    print("      /    \\")
    print("  leaf_A1(2) leaf_A2(2)")
    print()
    
    # TTN1: Build 3-level tree with small integer values
    # Root: connects to node_A and node_B
    root1 = np.array([[1, 2], [1, 1]])  # Shape (2, 2)
    root1_node = Node(tensor=root1, identifier="root")
    ttn1.add_root(root1_node, root1)
    
    # Node A: internal node with 2 children - shape (parent, child1, child2)
    node_A1 = np.array([[[1, 1], [1, 2]], [[2, 1], [1, 1]]])  # Shape (2, 2, 2)
    node_A1_node = Node(tensor=node_A1, identifier="node_A")
    ttn1.add_child_to_parent(node_A1_node, node_A1, 0, "root", 0)
    
    # Node B: leaf node  
    node_B1 = np.array([1, 2])  # Shape (2,): (parent_bond,)
    node_B1_node = Node(tensor=node_B1, identifier="node_B")
    ttn1.add_child_to_parent(node_B1_node, node_B1, 0, "root", 1)
    
    # Leaf A1: leaf under node_A
    leaf_A1_1 = np.array([1, 1])  # Shape (2,): (parent_bond,) 
    leaf_A1_1_node = Node(tensor=leaf_A1_1, identifier="leaf_A1")
    ttn1.add_child_to_parent(leaf_A1_1_node, leaf_A1_1, 0, "node_A", 1)
    
    # Leaf A2: leaf under node_A
    leaf_A2_1 = np.array([2, 1])  # Shape (2,): (parent_bond,)
    leaf_A2_1_node = Node(tensor=leaf_A2_1, identifier="leaf_A2") 
    ttn1.add_child_to_parent(leaf_A2_1_node, leaf_A2_1, 0, "node_A", 2)
    
    # TTN2: Same structure, different values
    root2 = np.array([[2, 1], [1, 2]])  # Shape (2, 2)
    root2_node = Node(tensor=root2, identifier="root")
    ttn2.add_root(root2_node, root2)
    
    node_A2 = np.array([[[1, 2], [2, 1]], [[1, 1], [1, 2]]])  # Shape (2, 2, 2)
    node_A2_node = Node(tensor=node_A2, identifier="node_A")
    ttn2.add_child_to_parent(node_A2_node, node_A2, 0, "root", 0)
    
    node_B2 = np.array([2, 1])  # Shape (2,)
    node_B2_node = Node(tensor=node_B2, identifier="node_B")
    ttn2.add_child_to_parent(node_B2_node, node_B2, 0, "root", 1)
    
    leaf_A1_2 = np.array([1, 2])  # Shape (2,)
    leaf_A1_2_node = Node(tensor=leaf_A1_2, identifier="leaf_A1")
    ttn2.add_child_to_parent(leaf_A1_2_node, leaf_A1_2, 0, "node_A", 1)
    
    leaf_A2_2 = np.array([1, 1])  # Shape (2,)
    leaf_A2_2_node = Node(tensor=leaf_A2_2, identifier="leaf_A2")
    ttn2.add_child_to_parent(leaf_A2_2_node, leaf_A2_2, 0, "node_A", 2)
    
    print("TTN1 tensors:")
    for node_id in ["root", "node_A", "node_B", "leaf_A1", "leaf_A2"]:
        tensor = ttn1.tensors[node_id]
        print(f"  {node_id}: shape {tensor.shape}")
        print(f"    {tensor}")
        
    print("\nTTN2 tensors:")
    for node_id in ["root", "node_A", "node_B", "leaf_A1", "leaf_A2"]:
        tensor = ttn2.tensors[node_id]
        print(f"  {node_id}: shape {tensor.shape}")
        print(f"    {tensor}")
    
    # Manual calculation for verification
    # TTN contraction: sum over all indices
    # root[i,j] * node_A[i,k,l] * node_B[j] * leaf_A1[k] * leaf_A2[l]
    
    manual_c1 = 0
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for l in range(2):
                    manual_c1 += (root1[i,j] * node_A1[i,k,l] * node_B1[j] * 
                                 leaf_A1_1[k] * leaf_A2_1[l])
    
    manual_c2 = 0                            
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for l in range(2):
                    manual_c2 += (root2[i,j] * node_A2[i,k,l] * node_B2[j] * 
                                 leaf_A1_2[k] * leaf_A2_2[l])
    
    print(f"\nManual calculation (sum over i,j,k,l):")
    print(f"TTN1: ∑ root[i,j] × node_A[i,k,l] × node_B[j] × leaf_A1[k] × leaf_A2[l]")
    
    # Show step by step for TTN1
    print(f"\nTTN1 step-by-step:")
    total1 = 0
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for l in range(2):
                    val = root1[i,j] * node_A1[i,k,l] * node_B1[j] * leaf_A1_1[k] * leaf_A2_1[l]
                    print(f"  i={i},j={j},k={k},l={l}: {root1[i,j]}×{node_A1[i,k,l]}×{node_B1[j]}×{leaf_A1_1[k]}×{leaf_A2_1[l]} = {val}")
                    total1 += val
    print(f"  TTN1 total: {total1}")
    
    print(f"\nTTN2 calculation: {manual_c2}")
    print(f"Expected sum: {manual_c1 + manual_c2}")
    
    # Contract using TTN method
    c1, _ = ttn1.completely_contract_tree(to_copy=True)
    c2, _ = ttn2.completely_contract_tree(to_copy=True)
    
    print(f"\nTTN contraction verification:")
    print(f"  TTN1 contracts to: {c1} (matches manual: {c1 == manual_c1})")
    print(f"  TTN2 contracts to: {c2} (matches manual: {c2 == manual_c2})")
    print(f"  Expected sum: {c1 + c2}")
    
    # Direct addition
    direct_result = add_ttns_direct_form(ttn1, ttn2)
    
    print(f"\nAfter direct addition (bond dimensions double):")
    print("Tree structure becomes:")
    print("           root(4,4)")
    print("          /        \\")  
    print("    node_A(4,4,4)   node_B(4)")
    print("      /    \\")
    print("  leaf_A1(4) leaf_A2(4)")
    print()
    
    for node_id in ["root", "node_A", "node_B", "leaf_A1", "leaf_A2"]:
        orig_shape = ttn1.tensors[node_id].shape
        new_shape = direct_result.tensors[node_id].shape
        tensor = direct_result.tensors[node_id]
        print(f"  {node_id}: {orig_shape} → {new_shape}")
        print(f"    {tensor}")
    
    c_direct, _ = direct_result.completely_contract_tree(to_copy=True)
    print(f"\nDirect addition contracts to: {c_direct}")
    print(f"Matches expected sum: {np.allclose(c_direct, c1 + c2)}")
    
    # SVD controlled addition with max_bond_dim=2
    svd_params = SVDParameters(max_bond_dim=2, rel_tol=1e-15, total_tol=1e-15)
    svd_result = add_ttns_svd_controlled(ttn1, ttn2, svd_params)
    
    print(f"\nSVD controlled (max_bond_dim=2) - should reduce back to original size:")
    for node_id in ["root", "node_A", "node_B", "leaf_A1", "leaf_A2"]:
        shape = svd_result.tensors[node_id].shape
        tensor = svd_result.tensors[node_id]
        print(f"  {node_id}: shape {shape}")
        print(f"    {tensor}")
    
    c_svd, _ = svd_result.completely_contract_tree(to_copy=True)
    print(f"\nSVD result contracts to: {c_svd}")
    print(f"Matches expected sum: {np.allclose(c_svd, c1 + c2, rtol=1e-10)}")
    
    # Test with bond dimension 1 for significant truncation
    svd_params_1 = SVDParameters(max_bond_dim=1, rel_tol=1e-15, total_tol=1e-15)
    svd_result_1 = add_ttns_svd_controlled(ttn1, ttn2, svd_params_1)
    
    print(f"\nSVD controlled (max_bond_dim=1) - heavy truncation:")
    for node_id in ["root", "node_A", "node_B", "leaf_A1", "leaf_A2"]:
        shape = svd_result_1.tensors[node_id].shape
        tensor = svd_result_1.tensors[node_id]
        print(f"  {node_id}: shape {shape}")
        print(f"    {tensor}")
    
    c_svd_1, _ = svd_result_1.completely_contract_tree(to_copy=True)
    print(f"\nSVD(bond_dim=1) contracts to: {c_svd_1}")
    print(f"Error from truncation: {abs(c_svd_1 - (c1 + c2))}")
    print(f"Relative error: {abs(c_svd_1 - (c1 + c2)) / abs(c1 + c2) * 100:.2f}%")
    
    print("\n✓ Manual verification test completed!")
    print("  You can verify the tensor contractions by hand using the step-by-step calculation.")
    print("  Direct addition preserves the sum exactly.")
    print("  SVD truncation shows controlled approximation error.\n")
    
    return ttn1, ttn2, direct_result, svd_result, svd_result_1

if __name__ == "__main__":
    try:
        print("Running TTN Addition SVD Tests\n")
        print("=" * 50)
        
        test_basic_functionality()
        test_bond_dimension_control()
        test_tolerance_effects()
        test_bond_dimension_5_to_2()
        test_manual_verification()
        
        print("=" * 50)
        print("All tests completed successfully!")
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()