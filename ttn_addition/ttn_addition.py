import numpy as np
from copy import deepcopy
from pytreenet.core.ttn import TreeTensorNetwork
from pytreenet.core.node import Node
from pytreenet.util.ttn_exceptions import NotCompatibleException

def _direct_tensor_addition(tensor1: np.ndarray, tensor2: np.ndarray, node1, node2) -> np.ndarray:
    """
    Individual tensor addition.
    """
    shape1 = tensor1.shape
    shape2 = tensor2.shape
    
    if len(shape1) != len(shape2):
        raise NotCompatibleException(
            f"Tensors must have same number of dimensions."
            f"Got {len(shape1)} vs {len(shape2)}"
        )
    
    if node1.is_leaf():
        if shape1[1:] != shape2[1:]:
            raise NotCompatibleException(
                f"Physical dimensions must match for leaf nodes. "
                f"Got {shape1[1:]} vs {shape2[1:]}"
            )
        
        new_parent_bond = shape1[0] + shape2[0]
        new_shape = (new_parent_bond,) + shape1[1:]
        
        new_tensor = np.zeros(new_shape, dtype=np.result_type(tensor1.dtype, tensor2.dtype))
        new_tensor[0:shape1[0]] = tensor1
        new_tensor[shape1[0]:] = tensor2
        
        return new_tensor
    
    else:

        num_children = len(node1.children)
        if node1.is_root():
            num_bond_dims = num_children
        else:
            num_bond_dims = 1 + num_children
        
        if len(node2.children) != num_children:
            raise NotCompatibleException(
                f"Nodes must have same number of children. Got {num_children} vs {len(node2.children)}"
            )
        
        if shape1[num_bond_dims:] != shape2[num_bond_dims:]:
            raise NotCompatibleException(
                f"Physical dimensions must match for internal nodes. "
                f"Got {shape1[num_bond_dims:]} vs {shape2[num_bond_dims:]}"
            )
        
        new_bond_dims = tuple(dim1 + dim2 for dim1, dim2 in zip(shape1[:num_bond_dims], shape2[:num_bond_dims]))
        new_shape = new_bond_dims + shape1[num_bond_dims:]
        
        new_tensor = np.zeros(new_shape, dtype=np.result_type(tensor1.dtype, tensor2.dtype))
        
        slices_A = tuple(slice(0, dim1) for dim1 in shape1[:num_bond_dims]) + tuple(slice(None) for _ in shape1[num_bond_dims:])
        new_tensor[slices_A] = tensor1
        slices_B = tuple(slice(dim1, dim1 + dim2) for dim1, dim2 in zip(shape1[:num_bond_dims], shape2[:num_bond_dims])) + tuple(slice(None) for _ in shape1[num_bond_dims:])
        new_tensor[slices_B] = tensor2
        
        return new_tensor

def add_ttns_direct_form(ttn1: TreeTensorNetwork, ttn2: TreeTensorNetwork) -> TreeTensorNetwork:
    """
    Adds two Tree Tensor Networks using direct form addition.
    
    Args:
        ttn1 (TreeTensorNetwork): First TTN to add
        ttn2 (TreeTensorNetwork): Second TTN to add
        
    Returns:
        TreeTensorNetwork: The sum of the two TTNs
        
    Raises:
        NotCompatibleException: If TTNs have incompatible structures
        ValueError: If TTNs are empty or have mismatched roots
    """

    _validate_structure(ttn1, ttn2)
    
    result_ttn = TreeTensorNetwork()
    
    direct_tensors = {}
    for node_id in ttn1.nodes:
        tensor1 = ttn1.tensors[node_id]
        tensor2 = ttn2.tensors[node_id]
        node1 = ttn1.nodes[node_id]
        node2 = ttn2.nodes[node_id]
        direct_tensors[node_id] = _direct_tensor_addition(tensor1, tensor2, node1, node2)
    
    root_id = ttn1.root_id
    root_tensor = direct_tensors[root_id]
    root_node = Node(tensor=root_tensor, identifier=root_id)
    result_ttn.add_root(root_node, root_tensor)
    
    nodes_to_process = [root_id]
    reconstructed_nodes = {root_id}
    
    while nodes_to_process:
        current_node_id = nodes_to_process.pop(0)
        current_node = ttn1.nodes[current_node_id]
        
        for child_id in current_node.children:
            if child_id not in reconstructed_nodes:
                child_tensor = direct_tensors[child_id]
                child_node = Node(tensor=child_tensor, identifier=child_id)
                
                child_node_orig = ttn1.nodes[child_id]
                parent_leg_in_child = child_node_orig.neighbour_index(current_node_id)
                child_leg_in_parent = current_node.neighbour_index(child_id)
                
                result_ttn.add_child_to_parent(
                    child_node,
                    child_tensor,
                    parent_leg_in_child,
                    current_node_id,
                    child_leg_in_parent
                )
                
                nodes_to_process.append(child_id)
                reconstructed_nodes.add(child_id)

    
    return result_ttn

def _validate_structure(ttn1: TreeTensorNetwork, ttn2: TreeTensorNetwork):
    """
    Validates that two TTNs have compatible structures for direct addition (i.e. the same 'shape').
    """
    if len(ttn1.nodes) != len(ttn2.nodes):
        raise NotCompatibleException(f"TTNs have different numbers of nodes: {len(ttn1.nodes)} vs {len(ttn2.nodes)}")
    
    if set(ttn1.nodes.keys()) != set(ttn2.nodes.keys()):
        raise NotCompatibleException("TTNs have different node identifiers")
    
    if ttn1.root_id != ttn2.root_id:
        raise NotCompatibleException(f"TTNs have different root nodes: {ttn1.root_id} vs {ttn2.root_id}")
    
    for node_id in ttn1.nodes:
        node1 = ttn1.nodes[node_id]
        node2 = ttn2.nodes[node_id]
        
        if node1.parent != node2.parent:
            raise NotCompatibleException(f"Node {node_id} has different parents in the two TTNs")
        if set(node1.children) != set(node2.children):
            raise NotCompatibleException(f"Node {node_id} has different children in the two TTNs")
        if set(node1.neighbouring_nodes()) != set(node2.neighbouring_nodes()):
            raise NotCompatibleException(f"Node {node_id} has different neighbor structure")