"""
Comparison of TTN Addition Methods: Direct vs Basis Extension

This script benchmarks and compares:
1. Direct addition (ttn_addition.py)
2. Basis extension addition (ttn_basis_extension_addition.py)

Generates plots showing:
- Number of nodes vs maximum bond dimension
- Number of nodes vs running time
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import psutil
import cProfile
import pstats
import io
from copy import deepcopy
from typing import List, Dict, Tuple

try:
    from pyinstrument import Profiler
    PYINSTRUMENT_AVAILABLE = True
except ImportError:
    PYINSTRUMENT_AVAILABLE = False

from ttn_addition import add_ttns_direct_form
from ttn_basis_extension_addition import ttn_basis_extension_addition
from ttn_addition_svd import add_ttns_svd_controlled
from addition_util import generate_tree_pair, generate_tree_pair_by_node_count, tree_statistics
from pytreenet.util.tensor_splitting import SVDParameters
from pytreenet.ttns.ttns import TreeTensorNetworkState
from pytreenet.contractions.tree_contraction import completely_contract_tree


def get_max_bond_dimension(ttn) -> int:
    """Get the maximum bond dimension in the tree."""
    max_bond = 0
    for node_id, tensor in ttn.tensors.items():
        node = ttn.nodes[node_id]

        # Get bond dimensions from parent and children legs
        bond_dims = []

        # Parent leg (if not root)
        if not node.is_root():
            bond_dims.append(tensor.shape[node.parent_leg])

        # Children legs (if not leaf)
        if not node.is_leaf():
            for child_leg in node.children_legs:
                bond_dims.append(tensor.shape[child_leg])

        # Update max bond dimension
        if bond_dims:
            max_bond = max(max_bond, max(bond_dims))

    return max_bond


def calculate_contraction_error(ttn1, ttn2) -> float:
    """
    Calculate the relative Frobenius norm error between two TTNs by contracting them.

    Args:
        ttn1: Reference TTN
        ttn2: Approximated TTN

    Returns:
        float: ||ttn1 - ttn2||_F / ||ttn1||_F
    """
    try:
        # Contract both trees
        contracted1 = completely_contract_tree(deepcopy(ttn1))
        contracted2 = completely_contract_tree(deepcopy(ttn2))

        # Extract tensors and orderings (handle tuple returns)
        ordering1 = None
        ordering2 = None

        if isinstance(contracted1, tuple):
            tensor1 = contracted1[0]
            ordering1 = contracted1[1]
        else:
            tensor1 = contracted1

        if isinstance(contracted2, tuple):
            tensor2 = contracted2[0]
            ordering2 = contracted2[1]
        else:
            tensor2 = contracted2

        # If orderings differ, transpose tensor2 to match tensor1's ordering
        if ordering1 is not None and ordering2 is not None and ordering1 != ordering2:
            # Create permutation to transform from ordering2 to ordering1
            perm = [ordering2.index(site) for site in ordering1]
            tensor2 = np.transpose(tensor2, perm)

        # Calculate relative error
        diff = tensor1 - tensor2
        error = np.linalg.norm(diff.flatten()) / np.linalg.norm(tensor1.flatten())

        return error

    except Exception as e:
        print(f"    Warning: Could not calculate contraction error: {e}")
        return None


def calculate_scalar_product_error(ttn1, ttn2) -> float:
    """
    Calculate error using TTNS scalar product

    Args:
        ttn1: Reference TTN
        ttn2: Approximated TTN

    Returns:
        float: Error measure based on normalized scalar product
    """
    try:
        # Cast to TTNS if needed
        if not isinstance(ttn1, TreeTensorNetworkState):
            ttns1 = TreeTensorNetworkState.__new__(TreeTensorNetworkState)
            ttns1.__dict__.update(ttn1.__dict__)
        else:
            ttns1 = ttn1

        if not isinstance(ttn2, TreeTensorNetworkState):
            ttns2 = TreeTensorNetworkState.__new__(TreeTensorNetworkState)
            ttns2.__dict__.update(ttn2.__dict__)
        else:
            ttns2 = ttn2


        # Compute scalar products
        overlap_12 = ttns1.scalar_product(ttns2, use_orthogonal_center=False)
        norm1_sq = ttns1.scalar_product(ttns1, use_orthogonal_center=False)
        norm2_sq = ttns2.scalar_product(ttns2, use_orthogonal_center=False)

        # Trace distance for pure states: D(ρ1, ρ2) = √(1 - |<ψ1|ψ2>|²/||ψ1||²||ψ2||²)
        # This accounts for both real and imaginary parts of the overlap
        norm1 = np.sqrt(np.abs(norm1_sq))
        norm2 = np.sqrt(np.abs(norm2_sq))
        normalized_overlap = np.abs(overlap_12) / (norm1 * norm2)

        # Trace distance (ranges from 0 to 1)
        error = np.sqrt(1 - normalized_overlap**2)

        # ttns1.normalise()
        # ttns2.normalise()
        
        # Calculate overlap between normalized states
        # overlap = ttns1.scalar_product(ttns2).real
        
        # Error formula: 2(1 - Re(<original|approximated>))
        # error = 2.0 * (1.0 - overlap)

        return max(0.0, error)  # Ensure non-negative

    except Exception as e:
        print(f"    Warning: Could not calculate scalar product error: {e}")
        import traceback
        traceback.print_exc()
        return None


def benchmark_method(tree1, tree2, method_func, method_name: str, num_runs: int = 3,
                     reference_result=None, error_method: str = 'contraction',
                     enable_profiling: bool = False, **method_kwargs) -> Dict:
    """Benchmark a single addition method.

    Args:
        error_method: 'contraction' or 'scalar_product' - method for error calculation
        enable_profiling: If True, profile the function and show top time consumers
    """
    times = []
    memory_peak = []
    profiler_obj = None

    for run in range(num_runs):
        # Create fresh copies for each run
        t1_copy = deepcopy(tree1)
        t2_copy = deepcopy(tree2)

        # Get memory before
        process = psutil.Process()
        mem_before = process.memory_info().rss / (1024**3)  # GB

        start_time = time.perf_counter()
        try:
            # Profile only the first run
            if enable_profiling and run == 0:
                if PYINSTRUMENT_AVAILABLE:
                    # Use pyinstrument for flame graph style profiling
                    profiler = Profiler()
                    profiler.start()
                    result = method_func(t1_copy, t2_copy, **method_kwargs)
                    profiler.stop()
                    profiler_obj = profiler
                else:
                    # Fall back to cProfile
                    profiler = cProfile.Profile()
                    profiler.enable()
                    result = method_func(t1_copy, t2_copy, **method_kwargs)
                    profiler.disable()
                    profiler_obj = profiler
            else:
                result = method_func(t1_copy, t2_copy, **method_kwargs)

            end_time = time.perf_counter()

            # Get memory after
            mem_after = process.memory_info().rss / (1024**3)  # GB
            mem_used = mem_after - mem_before

            times.append(end_time - start_time)
            memory_peak.append(mem_after)

        except Exception as e:
            print(f"  ERROR in {method_name}: {e}")
            return None

    # Get statistics from the result
    t1_final = deepcopy(tree1)
    t2_final = deepcopy(tree2)
    result = method_func(t1_final, t2_final, **method_kwargs)

    # Calculate result size
    result_memory = sys.getsizeof(result.tensors) / (1024**2)  # MB

    stats = tree_statistics(result)
    max_bond = get_max_bond_dimension(result)

    # Check for swapping
    swap = psutil.swap_memory()
    swap_used = swap.used / (1024**3) if swap.percent > 0 else 0  # GB

    # Calculate error if reference is provided
    error = None
    if reference_result is not None:
        if error_method == 'scalar_product':
            error = calculate_scalar_product_error(reference_result, result)
        else:  # default to contraction
            error = calculate_contraction_error(reference_result, result)

    result_dict = {
        'mean_time': np.mean(times),
        'std_time': np.std(times),
        'min_time': np.min(times),
        'max_time': np.max(times),
        'num_nodes': stats['total_nodes'],
        'max_bond': max_bond,
        'total_params': stats['total_parameters'],
        'error': error,
        'peak_memory_gb': np.max(memory_peak),
        'result_memory_mb': result_memory,
        'swap_used_gb': swap_used
    }

    # Save profiling results if enabled
    if enable_profiling and profiler_obj:
        safe_name = method_name.replace(' ', '_').replace('(', '').replace(')', '').replace('=', '')

        if PYINSTRUMENT_AVAILABLE:
            # Generate flame graph HTML
            html_file = f"profile_{safe_name}.html"
            with open(html_file, 'w') as f:
                f.write(profiler_obj.output_html())
            print(f"  Flame graph saved to {html_file}")

            # Also save text output
            txt_file = f"profile_{safe_name}.txt"
            with open(txt_file, 'w') as f:
                f.write(profiler_obj.output_text(unicode=True, color=False, show_all=False))
            print(f"  Text profiling saved to {txt_file}")
        else:
            # Save cProfile stats
            txt_file = f"profile_{safe_name}.txt"
            with open(txt_file, 'w') as f:
                ps = pstats.Stats(profiler_obj, stream=f)
                ps.sort_stats('cumulative')
                ps.print_stats(30)
            print(f"  Profiling saved to {txt_file}")

    return result_dict


def run_comparison(depths: List[int], branching_factor: int = 2,
                   leaf_probability: float = 0.2, num_runs: int = 3,
                   physical_dim: int = 2, bond_dim: int = 2) -> Dict:
    """Run comparison across multiple tree depths."""

    results = {
        'direct': [],
        'basis_extension': [],
        'depths': depths,
        'input_nodes': []
    }

    for depth in depths:
        print(f"\nTesting depth {depth}...")

        try:
            #Generate tree pair
            tree1, tree2 = generate_tree_pair(
                depth=depth,
                branching_factor=branching_factor,
                leaf_probability=leaf_probability,
                physical_dim=physical_dim,
                bond_dim=bond_dim
            )

            input_stats = tree_statistics(tree1)
            results['input_nodes'].append(input_stats['total_nodes'])
            print(f"  Input trees: {input_stats['total_nodes']} nodes")

            # Benchmark direct addition
            print("  Benchmarking direct addition...")
            direct_result = benchmark_method(tree1, tree2, add_ttns_direct_form,
                                            "Direct", num_runs)

            # Benchmark basis extension addition
            print("  Benchmarking basis extension addition...")
            basis_result = benchmark_method(tree1, tree2, ttn_basis_extension_addition,
                                           "Basis Extension", num_runs)

            results['direct'].append(direct_result)
            results['basis_extension'].append(basis_result)

            if direct_result and basis_result:
                print(f"  Direct: {direct_result['mean_time']:.4f}s, "
                      f"max_bond={direct_result['max_bond']}, "
                      f"nodes={direct_result['num_nodes']}")
                print(f"  Basis Ext: {basis_result['mean_time']:.4f}s, "
                      f"max_bond={basis_result['max_bond']}, "
                      f"nodes={basis_result['num_nodes']}")

        except Exception as e:
            print(f"  ERROR at depth {depth}: {e}")
            results['direct'].append(None)
            results['basis_extension'].append(None)
            results['input_nodes'].append(None)

    return results


def run_svd_truncation_comparison(depths: List[int],
                                   max_bond_dims: List[int],
                                   branching_factor: int = 2,
                                   leaf_probability: float = 0.2,
                                   num_runs: int = 3,
                                   physical_dim: int = 2,
                                   bond_dim: int = 2,
                                   svd_params_template: Dict = None,
                                   error_method: str = 'contraction') -> Dict:
    """Run comparison of basis extension with different SVD truncation parameters.

    Args:
        svd_params_template: Dictionary with SVD parameters (renorm, rel_tol, total_tol).
                           max_bond_dim will be set from max_bond_dims list.
        error_method: 'contraction' or 'scalar_product' - method for error calculation
    """

    results = {
        'depths': depths,
        'max_bond_dims': max_bond_dims,
        'input_nodes': [],
        'svd_results': {}  # Will be indexed by (depth_idx, max_bond_dim)
    }

    # Default SVD parameters if not provided
    if svd_params_template is None:
        svd_params_template = {
            'renorm': False,
            'rel_tol': 1e-12,
            'total_tol': 1e-12
        }

    for depth_idx, depth in enumerate(depths):
        print(f"\nTesting depth {depth}...")

        try:
            # Generate tree pair
            # tree1, tree2 = generate_tree_pair(
            #     depth=depth,
            #     branching_factor=branching_factor,
            #     leaf_probability=leaf_probability,
            #     physical_dim=physical_dim,
            #     bond_dim=bond_dim
            # )
            tree1, tree2 = generate_tree_pair_by_node_count(
                num_nodes=depth,
                physical_dim=physical_dim,
                bond_dim=bond_dim
            )

            input_stats = tree_statistics(tree1)
            results['input_nodes'].append(input_stats['total_nodes'])
            print(f"  Input trees: {input_stats['total_nodes']} nodes")

            # Compute reference (direct addition without truncation)
            print("  Computing reference (direct addition)...")
            t1_ref = deepcopy(tree1)
            t2_ref = deepcopy(tree2)
            reference = add_ttns_direct_form(t1_ref, t2_ref)

            # Test different max bond dimensions
            for max_bond in max_bond_dims:
                print(f"    Testing max_bond={max_bond}...")

                svd_params = SVDParameters(
                    max_bond_dim=max_bond,
                    **svd_params_template
                )

                result = benchmark_method(
                    tree1, tree2,
                    ttn_basis_extension_addition,
                    f"Basis Ext (max_bond={max_bond})",
                    num_runs=num_runs,
                    reference_result=reference,
                    error_method=error_method,
                    svd_params=svd_params
                )

                results['svd_results'][(depth_idx, max_bond)] = result

                if result:
                    error_str = f"{result['error']:.2e}" if result['error'] is not None else 'N/A'
                    print(f"      Time: {result['mean_time']:.4f}s, "
                          f"Max bond: {result['max_bond']}, "
                          f"Error: {error_str}")

        except Exception as e:
            print(f"  ERROR at depth {depth}: {e}")
            results['input_nodes'].append(None)
            for max_bond in max_bond_dims:
                results['svd_results'][(depth_idx, max_bond)] = None

    return results


def run_svd_method_comparison(depths: List[int],
                               max_bond_dims: List[int],
                               branching_factor: int = 2,
                               leaf_probability: float = 0.2,
                               num_runs: int = 3,
                               physical_dim: int = 2,
                               bond_dim: int = 2,
                               svd_params_template: Dict = None,
                               enable_profiling = False,
                               error_method: str = 'contraction') -> Dict:
    """Compare Direct+SVD vs Basis Extension+SVD against untruncated reference.

    Both methods use the same SVD parameters for fair comparison.
    """

    results = {
        'depths': depths,
        'max_bond_dims': max_bond_dims,
        'input_nodes': [],
        'svd_direct_results': {},  # Will be indexed by (depth_idx, max_bond_dim)
        'svd_basis_results': {}    # Will be indexed by (depth_idx, max_bond_dim)
    }

    # Default SVD parameters if not provided
    if svd_params_template is None:
        svd_params_template = {
            'renorm': False,
            'rel_tol': 1e-12,
            'total_tol': 1e-12
        }

    for depth_idx, depth in enumerate(depths):
        print(f"\nTesting depth {depth}...")

        try:
            # Generate tree pair
            tree1, tree2 = generate_tree_pair(
                depth=depth,
                branching_factor=branching_factor,
                leaf_probability=leaf_probability,
                physical_dim=physical_dim,
                bond_dim=bond_dim
            )

            input_stats = tree_statistics(tree1)
            results['input_nodes'].append(input_stats['total_nodes'])
            print(f"  Input trees: {input_stats['total_nodes']} nodes")

            # Compute reference (direct addition without truncation)
            print("  Computing reference (direct addition)...")
            t1_ref = deepcopy(tree1)
            t2_ref = deepcopy(tree2)
            reference = add_ttns_direct_form(t1_ref, t2_ref)

            # Test different max bond dimensions
            for max_bond in max_bond_dims:
                print(f"    Testing max_bond={max_bond}...")

                svd_params = SVDParameters(
                    max_bond_dim=max_bond,
                    **svd_params_template
                )

                # Benchmark Direct + SVD
                result_direct_svd = benchmark_method(
                    tree1, tree2,
                    add_ttns_svd_controlled,
                    f"Direct+SVD (max_bond={max_bond})",
                    num_runs=num_runs,
                    reference_result=reference,
                    error_method=error_method,
                    enable_profiling=enable_profiling,
                    svd_params=svd_params
                )

                # Benchmark Basis Extension + SVD
                result_basis_svd = benchmark_method(
                    tree1, tree2,
                    ttn_basis_extension_addition,
                    f"Basis+SVD (max_bond={max_bond})",
                    num_runs=num_runs,
                    reference_result=reference,
                    error_method=error_method,
                    enable_profiling=enable_profiling,
                    svd_params=svd_params
                )

                results['svd_direct_results'][(depth_idx, max_bond)] = result_direct_svd
                results['svd_basis_results'][(depth_idx, max_bond)] = result_basis_svd

                if result_direct_svd:
                    error_str = f"{result_direct_svd['error']:.2e}" if result_direct_svd['error'] is not None else 'N/A'
                    swap_str = f", Swap: {result_direct_svd['swap_used_gb']:.2f}GB" if result_direct_svd['swap_used_gb'] > 0 else ""
                    print(f"      Direct+SVD - Time: {result_direct_svd['mean_time']:.4f}s, "
                          f"Max bond: {result_direct_svd['max_bond']}, Error: {error_str}, "
                          f"Mem: {result_direct_svd['peak_memory_gb']:.2f}GB{swap_str}")

                if result_basis_svd:
                    error_str = f"{result_basis_svd['error']:.2e}" if result_basis_svd['error'] is not None else 'N/A'
                    swap_str = f", Swap: {result_basis_svd['swap_used_gb']:.2f}GB" if result_basis_svd['swap_used_gb'] > 0 else ""
                    print(f"      Basis+SVD  - Time: {result_basis_svd['mean_time']:.4f}s, "
                          f"Max bond: {result_basis_svd['max_bond']}, Error: {error_str}, "
                          f"Mem: {result_basis_svd['peak_memory_gb']:.2f}GB{swap_str}")

        except Exception as e:
            print(f"  ERROR at depth {depth}: {e}")
            results['input_nodes'].append(None)
            for max_bond in max_bond_dims:
                results['svd_direct_results'][(depth_idx, max_bond)] = None
                results['svd_basis_results'][(depth_idx, max_bond)] = None

    return results


def run_svd_method_comparison_by_nodes(node_counts: List[int],
                                        max_bond_dims: List[int],
                                        num_runs: int = 3,
                                        physical_dim: int = 2,
                                        bond_dim: int = 2,
                                        svd_params_template: Dict = None,
                                        error_method: str = 'contraction',
                                        enable_profiling: bool = False) -> Dict:
    """Compare Direct+SVD vs Basis Extension+SVD using node count instead of depth.

    Both methods use the same SVD parameters for fair comparison.
    """

    results = {
        'node_counts': node_counts,
        'max_bond_dims': max_bond_dims,
        'svd_direct_results': {},  # Will be indexed by (node_idx, max_bond_dim)
        'svd_basis_results': {}    # Will be indexed by (node_idx, max_bond_dim)
    }

    # Default SVD parameters if not provided
    if svd_params_template is None:
        svd_params_template = {
            'renorm': False,
            'rel_tol': 1e-12,
            'total_tol': 1e-12
        }

    for node_idx, num_nodes in enumerate(node_counts):
        print(f"\nTesting {num_nodes} nodes...")

        try:
            # Generate tree pair
            tree1, tree2 = generate_tree_pair_by_node_count(
                num_nodes=num_nodes,
                physical_dim=physical_dim,
                bond_dim=bond_dim
            )

            print(f"  Generated binary trees with {num_nodes} nodes")

            # Compute reference (direct addition without truncation)
            print("  Computing reference (direct addition)...")
            t1_ref = deepcopy(tree1)
            t2_ref = deepcopy(tree2)
            reference = add_ttns_direct_form(t1_ref, t2_ref)

            # Test different max bond dimensions
            for max_bond in max_bond_dims:
                print(f"    Testing max_bond={max_bond}...")

                svd_params = SVDParameters(
                    max_bond_dim=max_bond,
                    **svd_params_template
                )

                # Benchmark Direct + SVD
                result_direct_svd = benchmark_method(
                    tree1, tree2,
                    add_ttns_svd_controlled,
                    f"Direct+SVD (max_bond={max_bond})",
                    num_runs=num_runs,
                    reference_result=reference,
                    error_method=error_method,
                    enable_profiling=enable_profiling,
                    svd_params=svd_params
                )

                # Benchmark Basis Extension + SVD
                result_basis_svd = benchmark_method(
                    tree1, tree2,
                    ttn_basis_extension_addition,
                    f"Basis+SVD (max_bond={max_bond})",
                    num_runs=num_runs,
                    reference_result=reference,
                    error_method=error_method,
                    enable_profiling=enable_profiling,
                    svd_params=svd_params
                )

                results['svd_direct_results'][(node_idx, max_bond)] = result_direct_svd
                results['svd_basis_results'][(node_idx, max_bond)] = result_basis_svd

                if result_direct_svd:
                    error_str = f"{result_direct_svd['error']:.2e}" if result_direct_svd['error'] is not None else 'N/A'
                    swap_str = f", Swap: {result_direct_svd['swap_used_gb']:.2f}GB" if result_direct_svd['swap_used_gb'] > 0 else ""
                    print(f"      Direct+SVD - Time: {result_direct_svd['mean_time']:.4f}s, "
                          f"Max bond: {result_direct_svd['max_bond']}, Error: {error_str}, "
                          f"Mem: {result_direct_svd['peak_memory_gb']:.2f}GB{swap_str}")

                if result_basis_svd:
                    error_str = f"{result_basis_svd['error']:.2e}" if result_basis_svd['error'] is not None else 'N/A'
                    swap_str = f", Swap: {result_basis_svd['swap_used_gb']:.2f}GB" if result_basis_svd['swap_used_gb'] > 0 else ""
                    print(f"      Basis+SVD  - Time: {result_basis_svd['mean_time']:.4f}s, "
                          f"Max bond: {result_basis_svd['max_bond']}, Error: {error_str}, "
                          f"Mem: {result_basis_svd['peak_memory_gb']:.2f}GB{swap_str}")

        except Exception as e:
            print(f"  ERROR at {num_nodes} nodes: {e}")
            for max_bond in max_bond_dims:
                results['svd_direct_results'][(node_idx, max_bond)] = None
                results['svd_basis_results'][(node_idx, max_bond)] = None

    return results


def plot_results(results: Dict, save_prefix: str = "ttn_comparison"):
    """Generate comparison plots as separate figures."""

    # Extract valid data points
    depths = []
    input_nodes = []
    direct_times = []
    basis_times = []
    direct_bonds = []
    basis_bonds = []
    direct_nodes = []
    basis_nodes = []

    for i, depth in enumerate(results['depths']):
        if (results['direct'][i] is not None and
            results['basis_extension'][i] is not None and
            results['input_nodes'][i] is not None):

            depths.append(depth)
            input_nodes.append(results['input_nodes'][i])
            direct_times.append(results['direct'][i]['mean_time'])
            basis_times.append(results['basis_extension'][i]['mean_time'])
            direct_bonds.append(results['direct'][i]['max_bond'])
            basis_bonds.append(results['basis_extension'][i]['max_bond'])
            direct_nodes.append(results['direct'][i]['num_nodes'])
            basis_nodes.append(results['basis_extension'][i]['num_nodes'])

    if not depths:
        print("No valid data to plot!")
        return

    # Plot 1: Number of nodes vs Maximum bond dimension
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    ax1.plot(input_nodes, direct_bonds, 'o-', label='Direct Addition',
             color='blue', linewidth=2, markersize=8)
    ax1.plot(input_nodes, basis_bonds, 's-', label='Basis Extension',
             color='red', linewidth=2, markersize=8)
    ax1.set_xlabel('Number of Input Nodes', fontsize=14)
    ax1.set_ylabel('Maximum Bond Dimension', fontsize=14)
    ax1.set_title('Max Bond Dimension vs Tree Size', fontsize=15, fontweight='bold')
    ax1.legend(fontsize=12, loc='upper left', bbox_to_anchor=(1.02, 1))
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'pytreenet/core/addition/{save_prefix}_max_bond.png', dpi=300, bbox_inches='tight')
    print(f"Plot saved as pytreenet/core/addition/{save_prefix}_max_bond.png")
    plt.close()

    # Plot 2: Number of nodes vs Running time
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    ax2.plot(input_nodes, direct_times, 'o-', label='Direct Addition',
             color='blue', linewidth=2, markersize=8)
    ax2.plot(input_nodes, basis_times, 's-', label='Basis Extension',
             color='red', linewidth=2, markersize=8)
    ax2.set_xlabel('Number of Input Nodes', fontsize=14)
    ax2.set_ylabel('Running Time (seconds) (log scale)', fontsize=14)
    ax2.set_yscale('log')
    ax2.set_title('Running Time vs Tree Size', fontsize=15, fontweight='bold')
    ax2.legend(fontsize=12, loc='upper left', bbox_to_anchor=(1.02, 1))
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'pytreenet/core/addition/{save_prefix}_time.png', dpi=300, bbox_inches='tight')
    print(f"Plot saved as pytreenet/core/addition/{save_prefix}_time.png")
    plt.close()

    # Print summary table
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    print(f"{'Depth':<6} {'Nodes':<7} {'Direct Time':<13} {'Basis Time':<13} "
          f"{'Direct Bond':<12} {'Basis Bond':<12}")
    print("-"*80)

    for i in range(len(depths)):
        print(f"{depths[i]:<6} {input_nodes[i]:<7} "
              f"{direct_times[i]:<13.6f} {basis_times[i]:<13.6f} "
              f"{direct_bonds[i]:<12} {basis_bonds[i]:<12}")

    print("="*80)


def plot_svd_truncation_results(results: Dict, save_prefix: str = "ttn_svd_truncation"):
    """Generate plots for SVD truncation comparison as separate figures."""

    depths = results['depths']
    max_bond_dims = results['max_bond_dims']
    input_nodes = results['input_nodes']

    # Prepare data for plotting
    # For each max_bond_dim, collect error and time across depths
    data_by_bond = {max_bond: {'nodes': [], 'errors': [], 'times': [], 'actual_bonds': []}
                    for max_bond in max_bond_dims}

    for depth_idx, depth in enumerate(depths):
        if results['input_nodes'][depth_idx] is None:
            continue

        num_nodes = results['input_nodes'][depth_idx]

        for max_bond in max_bond_dims:
            result = results['svd_results'][(depth_idx, max_bond)]
            if result is not None:
                data_by_bond[max_bond]['nodes'].append(num_nodes)
                data_by_bond[max_bond]['errors'].append(result['error'] if result['error'] is not None else np.nan)
                data_by_bond[max_bond]['times'].append(result['mean_time'])
                data_by_bond[max_bond]['actual_bonds'].append(result['max_bond'])

    colors = plt.cm.viridis(np.linspace(0, 0.9, len(max_bond_dims)))
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p']

    # Plot 1: Error vs number of nodes
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    for idx, max_bond in enumerate(max_bond_dims):
        data = data_by_bond[max_bond]
        if data['nodes']:
            ax1.semilogy(data['nodes'], data['errors'],
                        marker=markers[idx % len(markers)],
                        label=f'max_bond={max_bond}',
                        color=colors[idx],
                        linewidth=2, markersize=8)

    ax1.set_xlabel('Number of Input Nodes', fontsize=14)
    ax1.set_ylabel('Relative Error (log scale)', fontsize=14)
    ax1.set_title('Relative Error vs Tree Size\n(vs Direct Addition)', fontsize=15, fontweight='bold')
    ax1.legend(fontsize=11, loc='upper left', bbox_to_anchor=(1.02, 1))
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'pytreenet/core/addition/{save_prefix}_error.png', dpi=300, bbox_inches='tight')
    print(f"Plot saved as pytreenet/core/addition/{save_prefix}_error.png")
    plt.close()

    # Plot 2: Running time vs number of nodes
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    for idx, max_bond in enumerate(max_bond_dims):
        data = data_by_bond[max_bond]
        if data['nodes']:
            ax2.plot(data['nodes'], data['times'],
                    marker=markers[idx % len(markers)],
                    label=f'max_bond={max_bond}',
                    color=colors[idx],
                    linewidth=2, markersize=8)

    ax2.set_xlabel('Number of Input Nodes', fontsize=14)
    ax2.set_ylabel('Running Time (seconds)', fontsize=14)
    ax2.set_title('Running Time vs Tree Size', fontsize=15, fontweight='bold')
    ax2.legend(fontsize=11, loc='upper left', bbox_to_anchor=(1.02, 1))
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'pytreenet/core/addition/{save_prefix}_time.png', dpi=300, bbox_inches='tight')
    print(f"Plot saved as pytreenet/core/addition/{save_prefix}_time.png")
    plt.close()

    # Plot 3: Actual max bond vs number of nodes
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    for idx, max_bond in enumerate(max_bond_dims):
        data = data_by_bond[max_bond]
        if data['nodes']:
            ax3.plot(data['nodes'], data['actual_bonds'],
                    marker=markers[idx % len(markers)],
                    label=f'max_bond={max_bond}',
                    color=colors[idx],
                    linewidth=2, markersize=8)

    ax3.set_xlabel('Number of Input Nodes', fontsize=14)
    ax3.set_ylabel('Actual Max Bond Dimension', fontsize=14)
    ax3.set_title('Actual Bond Dimension vs Tree Size', fontsize=15, fontweight='bold')
    ax3.legend(fontsize=11, loc='upper left', bbox_to_anchor=(1.02, 1))
    ax3.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'pytreenet/core/addition/{save_prefix}_bond.png', dpi=300, bbox_inches='tight')
    print(f"Plot saved as pytreenet/core/addition/{save_prefix}_bond.png")
    plt.close()

    # Print summary table
    print("\n" + "="*100)
    print("SVD TRUNCATION COMPARISON SUMMARY")
    print("="*100)
    print(f"{'Depth':<6} {'Nodes':<7} {'Max Bond':<10} {'Time (s)':<12} "
          f"{'Actual Bond':<12} {'Error':<12}")
    print("-"*100)

    for depth_idx, depth in enumerate(depths):
        if results['input_nodes'][depth_idx] is None:
            continue

        num_nodes = results['input_nodes'][depth_idx]

        for max_bond in max_bond_dims:
            result = results['svd_results'][(depth_idx, max_bond)]
            if result is not None:
                error_str = f"{result['error']:.2e}" if result['error'] is not None else "N/A"
                print(f"{depth:<6} {num_nodes:<7} {max_bond:<10} "
                      f"{result['mean_time']:<12.6f} {result['max_bond']:<12} {error_str:<12}")

    print("="*100)


def plot_svd_method_comparison_results(results: Dict, save_prefix: str = "ttn_svd_methods"):
    """Generate plots comparing Direct+SVD vs Basis Extension+SVD as separate figures."""

    # Check if using node_counts or depths
    if 'node_counts' in results:
        x_values = results['node_counts']
        x_label = 'Number of Nodes'
    else:
        x_values = results['depths']
        x_label = 'Depth'

    max_bond_dims = results['max_bond_dims']

    # Prepare data for plotting - separate for each method
    data_by_bond_direct = {max_bond: {'x': [], 'errors': [], 'times': [], 'actual_bonds': []}
                           for max_bond in max_bond_dims}
    data_by_bond_basis = {max_bond: {'x': [], 'errors': [], 'times': [], 'actual_bonds': []}
                          for max_bond in max_bond_dims}

    for idx, x_val in enumerate(x_values):
        for max_bond in max_bond_dims:
            result_direct = results['svd_direct_results'][(idx, max_bond)]
            result_basis = results['svd_basis_results'][(idx, max_bond)]

            if result_direct is not None:
                data_by_bond_direct[max_bond]['x'].append(x_val)
                data_by_bond_direct[max_bond]['errors'].append(result_direct['error'] if result_direct['error'] is not None else np.nan)
                data_by_bond_direct[max_bond]['times'].append(result_direct['mean_time'])
                data_by_bond_direct[max_bond]['actual_bonds'].append(result_direct['max_bond'])

            if result_basis is not None:
                data_by_bond_basis[max_bond]['x'].append(x_val)
                data_by_bond_basis[max_bond]['errors'].append(result_basis['error'] if result_basis['error'] is not None else np.nan)
                data_by_bond_basis[max_bond]['times'].append(result_basis['mean_time'])
                data_by_bond_basis[max_bond]['actual_bonds'].append(result_basis['max_bond'])

    colors = plt.cm.viridis(np.linspace(0, 0.9, len(max_bond_dims)))
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p']

    # Plot 1: Error comparison
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    for idx, max_bond in enumerate(max_bond_dims):
        data_direct = data_by_bond_direct[max_bond]
        data_basis = data_by_bond_basis[max_bond]

        if data_direct['x']:
            ax1.semilogy(data_direct['x'], data_direct['errors'],
                        marker=markers[idx % len(markers)],
                        label=f'Direct+SVD (max={max_bond})',
                        color=colors[idx],
                        linewidth=2, markersize=8, linestyle='-')

        if data_basis['x']:
            ax1.semilogy(data_basis['x'], data_basis['errors'],
                        marker=markers[idx % len(markers)],
                        label=f'Basis+SVD (max={max_bond})',
                        color=colors[idx],
                        linewidth=2, markersize=8, linestyle='--', alpha=0.7)

    ax1.set_xlabel(x_label, fontsize=14)
    ax1.set_ylabel('Error (log scale)', fontsize=14)
    ax1.set_title('Error Comparison: Direct+SVD vs Basis+SVD', fontsize=15, fontweight='bold')
    ax1.legend(fontsize=9, loc='upper left', bbox_to_anchor=(1.02, 1))
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'pytreenet/core/addition/{save_prefix}_error.png', dpi=300, bbox_inches='tight')
    print(f"Plot saved as pytreenet/core/addition/{save_prefix}_error.png")
    plt.close()

    # Plot 2: Time comparison
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    for idx, max_bond in enumerate(max_bond_dims):
        data_direct = data_by_bond_direct[max_bond]
        data_basis = data_by_bond_basis[max_bond]

        if data_direct['x']:
            ax2.plot(data_direct['x'], data_direct['times'],
                    marker=markers[idx % len(markers)],
                    label=f'Direct+SVD (max={max_bond})',
                    color=colors[idx],
                    linewidth=2, markersize=8, linestyle='-')

        if data_basis['x']:
            ax2.plot(data_basis['x'], data_basis['times'],
                    marker=markers[idx % len(markers)],
                    label=f'Basis+SVD (max={max_bond})',
                    color=colors[idx],
                    linewidth=2, markersize=8, linestyle='--', alpha=0.7)

    ax2.set_xlabel(x_label, fontsize=14)
    ax2.set_ylabel('Running Time (seconds) (log scale)', fontsize=14)
    ax2.set_title('Time Comparison: Direct+SVD vs Basis+SVD', fontsize=15, fontweight='bold')
    ax2.legend(fontsize=9, loc='upper left', bbox_to_anchor=(1.02, 1))
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    plt.tight_layout()
    plt.savefig(f'pytreenet/core/addition/{save_prefix}_time.png', dpi=300, bbox_inches='tight')
    print(f"Plot saved as pytreenet/core/addition/{save_prefix}_time.png")
    plt.close()


if __name__ == "__main__":
    print("TTN Addition Methods Comparison")
    print("=" * 80)

    # Choose comparison mode
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "basic"

    mode = "svd_nodes"

    if mode == "basic":
        # Basic comparison: Direct vs Basis Extension (no truncation)
        print("\n=== BASIC COMPARISON MODE ===\n")

        depths_to_test = list(range(1, 10))
        branching_factor = 1
        leaf_probability = 0
        physical_dim = 2
        bond_dim = 2
        num_runs = 3

        print(f"Parameters:")
        print(f"  Depths: {depths_to_test}")
        print(f"  Branching factor: {branching_factor}")
        print(f"  Leaf probability: {leaf_probability}")
        print(f"  Physical dimension: {physical_dim}")
        print(f"  Bond dimension: {bond_dim}")
        print(f"  Runs per test: {num_runs}")

        results = run_comparison(
            depths=depths_to_test,
            branching_factor=branching_factor,
            leaf_probability=leaf_probability,
            num_runs=num_runs,
            physical_dim=physical_dim,
            bond_dim=bond_dim
        )

        plot_results(results)

    elif mode == "svd":
        # SVD truncation comparison
        print("\n=== SVD TRUNCATION COMPARISON MODE ===\n")

        # Tree generation parameters
        depths_to_test =  list(range(2, 8))#[1, 2, 3] #, 4]#, 5, 6]#, 7] #list(range(1, 10)) 
        branching_factor = 1
        leaf_probability = 0
        physical_dim = 8
        bond_dim = 64
        num_runs = 3

        # SVD truncation parameters
        max_bond_dims_to_test = [64] #[8] #[64] # [4, 8, 16, 32, 64] #[32] #
        svd_params = {
            'renorm': False,
            'rel_tol': float("-inf"), # #  float("-inf")
            'total_tol': float("-inf") #, # float("-inf")
        }

        # Error calculation method: 'contraction' or 'scalar_product'
        error_method = 'contraction'

        print(f"Tree Parameters:")
        print(f"  Depths: {depths_to_test}")
        print(f"  Branching factor: {branching_factor}")
        print(f"  Leaf probability: {leaf_probability}")
        print(f"  Physical dimension: {physical_dim}")
        print(f"  Bond dimension: {bond_dim}")
        print(f"  Runs per test: {num_runs}")
        print(f"\nSVD Parameters:")
        print(f"  Max bond dimensions: {max_bond_dims_to_test}")
        print(f"  Renorm: {svd_params['renorm']}")
        print(f"  Relative tolerance: {svd_params['rel_tol']}")
        print(f"  Total tolerance: {svd_params['total_tol']}")
        print(f"\nError Calculation:")
        print(f"  Method: {error_method}")

        results = run_svd_truncation_comparison(
            depths=depths_to_test,
            max_bond_dims=max_bond_dims_to_test,
            branching_factor=branching_factor,
            leaf_probability=leaf_probability,
            num_runs=num_runs,
            physical_dim=physical_dim,
            bond_dim=bond_dim,
            svd_params_template=svd_params,
            error_method=error_method
        )

        plot_svd_truncation_results(results)

    elif mode == "svd_methods":
        # Compare Direct+SVD vs Basis Extension+SVD
        print("\n=== SVD METHODS COMPARISON MODE ===\n")

        # Tree generation parameters
        depths_to_test = list(range(1, 4))
        branching_factor = 2
        leaf_probability = 0
        physical_dim = 2
        bond_dim = 2
        num_runs = 5

        # SVD truncation parameters
        max_bond_dims_to_test = [64]
        svd_params = {
            'renorm': False,
            'rel_tol': float("-inf"),
            'total_tol': 1e-15 # float("-inf") #  #
        }

        enable_profiling = False
        # Error calculation method: 'contraction' or 'scalar_product'
        error_method = 'contraction'

        print(f"Tree Parameters:")
        print(f"  Depths: {depths_to_test}")
        print(f"  Branching factor: {branching_factor}")
        print(f"  Leaf probability: {leaf_probability}")
        print(f"  Physical dimension: {physical_dim}")
        print(f"  Bond dimension: {bond_dim}")
        print(f"  Runs per test: {num_runs}")
        print(f"\nSVD Parameters:")
        print(f"  Max bond dimensions: {max_bond_dims_to_test}")
        print(f"  Renorm: {svd_params['renorm']}")
        print(f"  Relative tolerance: {svd_params['rel_tol']}")
        print(f"  Total tolerance: {svd_params['total_tol']}")
        print(f"\nError Calculation:")
        print(f"  Method: {error_method}")

        results = run_svd_method_comparison(
            depths=depths_to_test,
            max_bond_dims=max_bond_dims_to_test,
            branching_factor=branching_factor,
            leaf_probability=leaf_probability,
            num_runs=num_runs,
            physical_dim=physical_dim,
            bond_dim=bond_dim,
            svd_params_template=svd_params,
            error_method=error_method,
            enable_profiling=enable_profiling
        )

        plot_svd_method_comparison_results(results)

    elif mode == "svd_nodes":
        # Compare Direct+SVD vs Basis Extension+SVD using node counts
        print("\n=== SVD METHODS COMPARISON BY NODE COUNT ===\n")

        # Tree generation parameters
        node_counts_to_test = list(range(2, 20)) # [20] #
        num_runs = 3
        physical_dim = 2
        bond_dim = 2

        # SVD truncation parameters
        max_bond_dims_to_test = [64] #[4, 8, 16, 32]
        svd_params = {
            'renorm': False,
            'rel_tol': float("-inf"),
            'total_tol': 1e-1, # float("-inf"), # 
        }

        # Error calculation method: 'contraction' or 'scalar_product'
        error_method = 'contraction'

        # Enable profiling to see which functions take the most time
        enable_profiling = False  # Set to False to disable profiling

        print(f"Tree Parameters:")
        print(f"  Node counts: {node_counts_to_test}")
        print(f"  Physical dimension: {physical_dim}")
        print(f"  Bond dimension: {bond_dim}")
        print(f"  Runs per test: {num_runs}")
        print(f"\nSVD Parameters:")
        print(f"  Max bond dimensions: {max_bond_dims_to_test}")
        print(f"  Renorm: {svd_params['renorm']}")
        print(f"  Relative tolerance: {svd_params['rel_tol']}")
        print(f"  Total tolerance: {svd_params['total_tol']}")
        print(f"\nError Calculation:")
        print(f"  Method: {error_method}")

        results = run_svd_method_comparison_by_nodes(
            node_counts=node_counts_to_test,
            max_bond_dims=max_bond_dims_to_test,
            num_runs=num_runs,
            physical_dim=physical_dim,
            bond_dim=bond_dim,
            svd_params_template=svd_params,
            error_method=error_method,
            enable_profiling=enable_profiling
        )

        plot_svd_method_comparison_results(results, save_prefix="ttn_svd_methods_nodes")

    else:
        print(f"Unknown mode: {mode}")
        print("Usage: python compare_addition_methods.py [basic|svd|svd_methods|svd_nodes]")
        print("  basic       - Compare direct addition vs basis extension (no truncation)")
        print("  svd         - Compare basis extension with different SVD truncation parameters")
        print("  svd_methods - Compare Direct+SVD vs Basis Extension+SVD (by depth)")
        print("  svd_nodes   - Compare Direct+SVD vs Basis Extension+SVD (by node count)")
