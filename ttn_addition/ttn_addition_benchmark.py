import numpy as np
import time
from typing import Tuple, List, Dict, Any
from copy import deepcopy
from pytreenet.core.ttn import TreeTensorNetwork
from pytreenet.ttns.ttns import TreeTensorNetworkState
from pytreenet.random.random_node import random_tensor_node
from pytreenet.random.random_ttns import RandomTTNSMode
from pytreenet.util.tensor_splitting import SVDParameters


def generate_tree_pair(depth: int, branching_factor: int = 2, leaf_probability: float = 0.2,
                      physical_dim: int = 2, bond_dim: int = 2) -> Tuple[TreeTensorNetwork, TreeTensorNetwork]:
    """
    Generate two structurally identical TTN trees with different random tensor values.

    Uses probabilistic early termination to create irregular tree structures.

    Args:
        depth (int): Maximum depth of the tree (0 = root only, 1 = root + children, etc.)
        branching_factor (int): Number of children per internal node. Defaults to 2.
        leaf_probability (float): Probability that a node becomes a leaf early. Defaults to 0.2.
        physical_dim (int): Physical dimension for each node. Defaults to 2.
        bond_dim (int): Virtual bond dimension for tensor legs. Defaults to 2.

    Returns:
        Tuple[TreeTensorNetwork, TreeTensorNetwork]: Two structurally identical trees with different tensors
    """    
    if depth < 0:
        raise ValueError("Depth must be non-negative")
    
    ttn1 = TreeTensorNetwork()
    ttn2 = TreeTensorNetwork()

    if depth == 0:
        root_node1, root_tensor1 = random_tensor_node((physical_dim,), identifier="root")
        root_node2, root_tensor2 = random_tensor_node((physical_dim,), identifier="root")
        ttn1.add_root(root_node1, root_tensor1)
        ttn2.add_root(root_node2, root_tensor2)
        return ttn1, ttn2

    root_shape = tuple([bond_dim] * branching_factor + [physical_dim])
    root_node1, root_tensor1 = random_tensor_node(root_shape, identifier="root")
    root_node2, root_tensor2 = random_tensor_node(root_shape, identifier="root")
    ttn1.add_root(root_node1, root_tensor1)
    ttn2.add_root(root_node2, root_tensor2)
    
    current_level = [("root", list(range(branching_factor)))]
    node_counter = 0
    
    for level in range(1, depth + 1):
        next_level = []
        
        for parent_id, parent_legs in current_level:
            for leg_idx in parent_legs:
                # Determine if this should be a leaf (either at max depth or early termination)
                is_leaf = (level == depth) or (level < depth and np.random.random() < leaf_probability)
                
                node_counter += 1
                child_id = f"node_{node_counter}"
                
                # Create appropriate tensor shape based on whether it's actually a leaf
                if is_leaf:
                    child_shape = (bond_dim, physical_dim)  # parent leg + physical leg only
                else:
                    child_shape = tuple([bond_dim] * (1 + branching_factor) + [physical_dim])  # parent + children + physical
                    next_level.append((child_id, list(range(1, branching_factor + 1))))
                
                # Create nodes for both trees with same structure but different tensors
                child_node1, child_tensor1 = random_tensor_node(child_shape, identifier=child_id)
                child_node2, child_tensor2 = random_tensor_node(child_shape, identifier=child_id)
                
                ttn1.add_child_to_parent(child_node1, child_tensor1, 0, parent_id, leg_idx)
                ttn2.add_child_to_parent(child_node2, child_tensor2, 0, parent_id, leg_idx)
        
        current_level = next_level
    
    return ttn1, ttn2


def generate_tree_pair_by_node_count(num_nodes: int,
                                     physical_dim: int = 2,
                                     bond_dim: int = 2) -> Tuple[TreeTensorNetwork, TreeTensorNetwork]:
    """
    Generate two structurally identical binary TTN trees with a target number of nodes.

    Builds trees level-by-level: root at level 0, then its 2 children at level 1,
    then up to 4 nodes at level 2, etc. Structure: 0(1(3,4), 2(5,6), ...)

    Args:
        num_nodes (int): Target number of nodes in the tree
        physical_dim (int): Physical dimension for each node. Defaults to 2.
        bond_dim (int): Virtual bond dimension for tensor legs. Defaults to 2.

    Returns:
        Tuple[TreeTensorNetwork, TreeTensorNetwork]: Two structurally identical trees with different tensors
    """
    if num_nodes < 1:
        raise ValueError("Number of nodes must be at least 1")

    ttn1 = TreeTensorNetwork()
    ttn2 = TreeTensorNetwork()

    if num_nodes == 1:
        # Single root node
        root_node1, root_tensor1 = random_tensor_node((physical_dim,), identifier="root")
        root_node2, root_tensor2 = random_tensor_node((physical_dim,), identifier="root")
        ttn1.add_root(root_node1, root_tensor1)
        ttn2.add_root(root_node2, root_tensor2)
        return ttn1, ttn2

    # Create root with space for 2 children (binary tree)
    root_shape = tuple([bond_dim] * 2 + [physical_dim])
    root_node1, root_tensor1 = random_tensor_node(root_shape, identifier="root")
    root_node2, root_tensor2 = random_tensor_node(root_shape, identifier="root")
    ttn1.add_root(root_node1, root_tensor1)
    ttn2.add_root(root_node2, root_tensor2)

    nodes_created = 1
    node_counter = 0

    # Binary tree - level-by-level construction
    # Current level stores tuples of (parent_id, parent_leg_index)
    current_level = [("root", 0), ("root", 1)]

    while nodes_created < num_nodes and current_level:
        next_level = []

        for parent_id, parent_leg in current_level:
            if nodes_created >= num_nodes:
                break

            node_counter += 1
            child_id = f"node_{node_counter}"
            nodes_created += 1

            # Calculate what the children indices would be
            # For node i, its children would be at indices 2*i+1 and 2*i+2
            left_child_idx = 2 * node_counter + 1
            right_child_idx = 2 * node_counter + 2

            # Determine how many children this node has based on which indices are within range
            has_left_child = left_child_idx <= num_nodes - 1  # -1 because root is node 0
            has_right_child = right_child_idx <= num_nodes - 1

            num_children = int(has_left_child) + int(has_right_child)

            if num_children == 2:
                child_shape = tuple([bond_dim] * 3 + [physical_dim])  # parent + 2 children + physical
                next_level.append((child_id, 1))
                next_level.append((child_id, 2))
            elif num_children == 1:
                child_shape = tuple([bond_dim] * 2 + [physical_dim])  # parent + 1 child + physical
                next_level.append((child_id, 1))
            else:
                child_shape = (bond_dim, physical_dim)  # parent + physical (leaf)

            child_node1, child_tensor1 = random_tensor_node(child_shape, identifier=child_id)
            child_node2, child_tensor2 = random_tensor_node(child_shape, identifier=child_id)

            ttn1.add_child_to_parent(child_node1, child_tensor1, 0, parent_id, parent_leg)
            ttn2.add_child_to_parent(child_node2, child_tensor2, 0, parent_id, parent_leg)

        current_level = next_level

    return ttn1, ttn2


def tree_statistics(ttn: TreeTensorNetwork) -> dict:
    """
    Args:
        ttn (TreeTensorNetwork): The tree to analyze
        
    Returns:
        dict: Statistics including node count, depth, parameters, etc.
    """
    if not ttn.nodes:
        return {"total_nodes": 0, "leaf_nodes": 0, "internal_nodes": 0, 
                "depth": 0, "total_parameters": 0}
    
    stats = {
        "total_nodes": len(ttn.nodes),
        "leaf_nodes": 0,
        "internal_nodes": 0,
        "depth": 0,
        "total_parameters": 0
    }
    
    queue = [(ttn.root_id, 0)]
    visited = set()
    
    while queue:
        node_id, level = queue.pop(0)
        if node_id in visited:
            continue
        visited.add(node_id)
        
        stats["depth"] = max(stats["depth"], level)
        node = ttn.nodes[node_id]
        tensor = ttn.tensors[node_id]
        
        stats["total_parameters"] += np.prod(tensor.shape)
        
        if node.is_leaf():
            stats["leaf_nodes"] += 1
        else:
            stats["internal_nodes"] += 1
            for child_id in node.children:
                queue.append((child_id, level + 1))
    
    return stats


def calculate_squared_error(ttn1: TreeTensorNetwork, ttn2: TreeTensorNetwork) -> float:
    """
    Calculate the squared error between two normalized states. Error = 2(1 - Re(<original|approximated>)
        
    Returns:
        float: The squared error between the two trees
    """
    try:
        # Cast TTNs to TTNSs using the original method that worked
        ttns1 = TreeTensorNetworkState.__new__(TreeTensorNetworkState)
        ttns1.__dict__.update(ttn1.__dict__)
        
        ttns2 = TreeTensorNetworkState.__new__(TreeTensorNetworkState)
        ttns2.__dict__.update(ttn2.__dict__)
        
        # Normalize both states in place using the TTNS normalize method
        ttns1.norm()
        ttns2.norm()
        
        # Calculate overlap between normalized states
        overlap = ttns1.scalar_product(ttns2).real
        
        # Error formula: 2(1 - Re(<original|approximated>))
        error = 2.0 * (1.0 - overlap)
        
        return max(0.0, error)  # Ensure non-negative
        
    except Exception as e:
        print(f"\tERROR calculating the squared error: {e}")
        return float('inf')


def benchmark_addition_speed(tree1: TreeTensorNetwork, tree2: TreeTensorNetwork, 
                           addition_func, num_runs: int = 5) -> dict:
    """
    Benchmark the speed of TTN addition between two trees.
    
    Args:
        tree1 (TreeTensorNetwork): First tree for addition
        tree2 (TreeTensorNetwork): Second tree for addition  
        addition_func: The addition function to benchmark
        num_runs (int): Number of runs for averaging. Defaults to 5.
        
    Returns:
        dict: Timing statistics including mean, std, min, max, and all the times measured
    """
    times = []
    
    for _ in range(num_runs):
        start_time = time.perf_counter()
        try:
            result = addition_func(tree1, tree2)
            end_time = time.perf_counter()
            times.append(end_time - start_time)
        except Exception as e:
            return {"error": str(e), "mean_time": None, "std_time": None}
    
    return {
        "mean_time": np.mean(times),
        "std_time": np.std(times),
        "min_time": np.min(times),
        "max_time": np.max(times),
        "all_times": times
    }


def benchmark_svd_accuracy(tree1: TreeTensorNetwork, tree2: TreeTensorNetwork,
                          direct_addition_func, svd_addition_func,
                          svd_params_list: List[Dict[str, Any]],
                          num_runs: int = 3) -> Dict[str, Any]:
    """
    Benchmark SVD truncation accuracy for different parameters.
    
    Args:
        tree1 (TreeTensorNetwork): First tree for addition
        tree2 (TreeTensorNetwork): Second tree for addition
        direct_addition_func: Direct addition function (no truncation)
        svd_addition_func: SVD addition function
        svd_params_list (List[Dict[str, Any]]): List of SVD parameter dictionaries
        num_runs (int): Number of runs for averaging. Defaults to 3.
        
    Returns:
        Dict[str, Any]: Results containing accuracy and timing for each parameter set
    """
    # Compute reference (exact) result
    reference_result = direct_addition_func(tree1, tree2)
    
    results = {
        "reference_stats": tree_statistics(reference_result),
        "parameter_sweep": []
    }
    
    for params in svd_params_list:
        param_results = {
            "parameters": params,
            "errors": [],
            "times": [],
            "approximated_stats": None
        }
        
        for run in range(num_runs):
            try:
                start_time = time.perf_counter()
                
                # Create SVDParameters object
                svd_params = SVDParameters(
                    max_bond_dim=params.get("max_bond_dim", 100),
                    rel_tol=params.get("rel_tol", 1e-12),
                    total_tol=params.get("total_tol", 1e-12),
                    renorm=params.get("renorm", False),
                    sum_trunc=params.get("sum_trunc", False)
                )
                
                approximated_result = svd_addition_func(tree1, tree2, svd_params)
                end_time = time.perf_counter()
                
                # Calculate squared error
                error = calculate_squared_error(reference_result, approximated_result)
                
                param_results["errors"].append(error)
                param_results["times"].append(end_time - start_time)
                
                if run == 0:  # Store stats from first run
                    param_results["approximated_stats"] = tree_statistics(approximated_result)
                    
            except Exception as e:
                param_results["error"] = str(e)
                break
        
        if param_results["errors"]:
            param_results["mean_error"] = np.mean(param_results["errors"])
            param_results["std_error"] = np.std(param_results["errors"])
            param_results["mean_time"] = np.mean(param_results["times"])
            param_results["std_time"] = np.std(param_results["times"])
        
        results["parameter_sweep"].append(param_results)
    
    return results


if __name__ == "__main__":
    try:
        from ttn_addition import add_ttns_direct_form
        from ttn_addition_svd import add_ttns_svd_controlled

    except ImportError:
        print("Warning: Could not import ttn_addition.add_ttns_direct_form function")
        add_ttns_direct_form = None
        add_ttns_svd_controlled = None

    print("TTN Addition Benchmarking Suite")
    
    # Define SVD parameter sets to test - using more conservative values
    svd_parameter_sets = [
        {"max_bond_dim": 20, "rel_tol": 1e-6, "total_tol": 1e-6},
        {"max_bond_dim": 20, "rel_tol": 1e-8, "total_tol": 1e-8},
        {"max_bond_dim": 50, "rel_tol": 1e-6, "total_tol": 1e-6},
        {"max_bond_dim": 50, "rel_tol": 1e-8, "total_tol": 1e-8},
        {"max_bond_dim": 100, "rel_tol": 1e-6, "total_tol": 1e-6},
        {"max_bond_dim": 100, "rel_tol": 1e-8, "total_tol": 1e-8},
    ]
    
    depths_to_test = [10, 11, 12, 13, 14, 15]
    
    for depth in depths_to_test:
        print(f"\nTesting depth {depth}:")
        
        try:
            tree1, tree2 = generate_tree_pair(depth=depth, leaf_probability=0.2)            
            stats1 = tree_statistics(tree1)
            
            print(f"\tTrees: {stats1['total_nodes']} nodes, {stats1['total_parameters']} parameters")
            
            #Test direct addition
            if add_ttns_direct_form is not None:
                print("\tBenchmarking addition:")
                timing_results = benchmark_addition_speed(tree1, tree2, add_ttns_direct_form)
                
                #print direct addition result
                if "error" in timing_results:
                    print(f"\tERROR: {timing_results['error']}")
                else:
                    print(f"\tAddition time: {timing_results['mean_time']:.6f}s ± {timing_results['std_time']:.6f}s")
                    print(f"\tRange: {timing_results['min_time']:.6f}s - {timing_results['max_time']:.6f}s")
                
                # Test SVD accuracy 
                if add_ttns_svd_controlled is not None and depth <= 11:
                    print(f"\tSVD Accuracy Analysis for depth {depth}:")
                    svd_results = benchmark_svd_accuracy(tree1, tree2, add_ttns_direct_form, 
                                                       add_ttns_svd_controlled, svd_parameter_sets[:4])  # Test first 4 parameter sets
                    
                    #print SVD accuracy results 
                    if "error" in svd_results:
                        print(f"\t\tERROR: {svd_results['error']}")
                    else:
                        ref_stats = svd_results["reference_stats"]
                        print(f"\t\tReference: {ref_stats['total_nodes']} nodes, {ref_stats['total_parameters']} parameters")
                        
                        print(f"\t\t{'Max Bond':<8} {'Tol':<8} {'Mean Error':<12} {'Time (s)':<8} {'Nodes':<6}")
                        print(f"\t\t{'-'*50}")
                        
                        for result in svd_results["parameter_sweep"]:
                            if "error" in result:
                                print(f"\t\tERROR with {result['parameters']}: {result['error']}")
                                continue
                                
                            params = result["parameters"]
                            approx_stats = result["approximated_stats"]
                            
                            print(f"\t\t{params['max_bond_dim']:<8} "
                                  f"{params['rel_tol']:<8.0e} "
                                  f"{result['mean_error']:<12.2e} "
                                  f"{result['mean_time']:<8.3f} "
                                  f"{approx_stats['total_nodes']:<6}")
                
        except Exception as e:
            print(f"\tERROR generating trees at depth {depth}: {e}")
            continue