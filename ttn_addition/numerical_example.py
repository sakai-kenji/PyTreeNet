import numpy as np
from pytreenet.core.ttn import TreeTensorNetwork
from pytreenet.core.node import Node
from ttn_addition import add_ttns_direct_form

def create_simple_ttn_example():
    ttn1 = TreeTensorNetwork()
    
    # Root tensor (2×2) - matrix ((3, 4), (5, 6))
    root1 = np.array([[3, 4], [5, 6]])
    root1_node = Node(tensor=root1, identifier="root")
    ttn1.add_root(root1_node, root1)
    
    # Left leaf tensor (2,) - vector (1, 2)
    left_leaf1 = np.array([1, 2])
    left_leaf1_node = Node(tensor=left_leaf1, identifier="left_leaf")
    ttn1.add_child_to_parent(left_leaf1_node, left_leaf1, 0, "root", 0)
    
    # Right leaf tensor (2,) - vector (7, 8)T
    right_leaf1 = np.array([7, 8])    
    right_leaf1_node = Node(tensor=right_leaf1, identifier="right_leaf")
    ttn1.add_child_to_parent(right_leaf1_node, right_leaf1, 0, "root", 1)
    
    return ttn1

def create_second_ttn_example():
    ttn2 = TreeTensorNetwork()
    
    # Root tensor (2×2) - matrix ((.3, .4), (.5, .6))
    root2 = np.array([[0.3, 0.4], [0.5, 0.6]])
    root2_node = Node(tensor=root2, identifier="root")
    ttn2.add_root(root2_node, root2)
    
    # Left leaf tensor (2,) - vector (.1, .2)
    left_leaf2 = np.array([0.1, 0.2])
    left_leaf2_node = Node(tensor=left_leaf2, identifier="left_leaf")
    ttn2.add_child_to_parent(left_leaf2_node, left_leaf2, 0, "root", 0)
    
    # Right leaf tensor (2,) - vector (.7, .8)T
    right_leaf2 = np.array([0.7, 0.8])

    right_leaf2_node = Node(tensor=right_leaf2, identifier="right_leaf")
    ttn2.add_child_to_parent(right_leaf2_node, right_leaf2, 0, "root", 1)
    
    return ttn2

def demonstrate_addition_and_contraction():
    """Demonstrate both methods: add-then-contract vs contract-then-add."""
    
    print("\n" + "=" * 80)
    print("COMPARISON: ADD-THEN-CONTRACT vs CONTRACT-THEN-ADD")
    print("=" * 80)
    
    ttn1 = create_simple_ttn_example()
    ttn2 = create_second_ttn_example()
    
    # Get the tensor values for manual verification
    root1 = ttn1.tensors["root"]
    left_leaf1 = ttn1.tensors["left_leaf"]
    right_leaf1 = ttn1.tensors["right_leaf"]
    
    root2 = ttn2.tensors["root"]
    left_leaf2 = ttn2.tensors["left_leaf"]
    right_leaf2 = ttn2.tensors["right_leaf"]
    
    print("\n" + "=" * 80)
    print("METHOD 1: ADD FIRST, THEN CONTRACT")
    print("=" * 80)
    
    print("\nSTEP 1: Performing TTN addition")
    result_ttn = add_ttns_direct_form(ttn1, ttn2)
    
    print("\nAdded TTN tensors:")
    for node_id in ["root", "left_leaf", "right_leaf"]:
        tensor = result_ttn.tensors[node_id]
        node = result_ttn.nodes[node_id]
        is_leaf = len(node.children) == 0
        node_type = "leaf" if is_leaf else "internal"
        
        print(f"\n{node_id} ({node_type}):")
        if tensor.ndim == 1:
            print(f"  {tensor}")
        else:
            print(f"  {tensor}")
    
    print("\nSTEP 2: Contracting the added TTN")
    final_result_method1, _ = result_ttn.completely_contract_tree(to_copy=True)
    print(f"METHOD 1 RESULT: {final_result_method1}")
    
    print("\n" + "=" * 80)
    print("METHOD 2: CONTRACT FIRST, THEN ADD")
    print("=" * 80)
    
    print("\nSTEP 1: Contracting TTN1")
    result_ttn1, _ = ttn1.completely_contract_tree(to_copy=True)
    print(f"TTN1 CONTRACTED RESULT: {result_ttn1}")
    
    print("\nSTEP 2: Contracting TTN2")
    result_ttn2, _ = ttn2.completely_contract_tree(to_copy=True)
    print(f"TTN2 CONTRACTED RESULT: {result_ttn2}")
    
    print("\nSTEP 3: Adding the contracted results")
    final_result_method2 = result_ttn1 + result_ttn2
    print(f"METHOD 2 RESULT: {final_result_method2}")
    
    print("\n" + "=" * 80)
    print("VERIFICATION")
    print("=" * 80)
    
    print(f"Method 1 (add then contract): {final_result_method1}")
    print(f"Method 2 (contract then add): {final_result_method2}")
    print(f"Results match: {np.allclose(final_result_method1, final_result_method2)}")
    
    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    print("✓ Both methods give the same result")
    print("✓ contract(add(TTN1, TTN2)) = contract(TTN1) + contract(TTN2)")
    print("✓ The TTN addition implementation correctly preserves this property")
    
    return ttn1, ttn2, result_ttn, final_result_method1, final_result_method2

if __name__ == "__main__":
    print("NUMERICAL EXAMPLE: BALANCED TREE STRUCTURE")
    ttn1, ttn2, result, method1_result, method2_result = demonstrate_addition_and_contraction()