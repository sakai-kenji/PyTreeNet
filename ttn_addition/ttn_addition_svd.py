from copy import deepcopy
from pytreenet.core.ttn import TreeTensorNetwork
from pytreenet.util.tensor_splitting import SVDParameters
from pytreenet.core.truncation.recursive_truncation import recursive_truncation
from ttn_addition import add_ttns_direct_form

def add_ttns_svd_controlled(ttn1: TreeTensorNetwork, 
                            ttn2: TreeTensorNetwork, 
                            svd_params: SVDParameters = SVDParameters()
                            ) -> TreeTensorNetwork:
    """
    Adds two Tree Tensor Networks using direct form addition followed by SVD truncation
    to manage bond dimension growth.

    Args:
        ttn1 (TreeTensorNetwork): First TTN to add
        ttn2 (TreeTensorNetwork): Second TTN to add
        svd_params (SVDParameters, optional): Parameters controlling SVD truncation.
            Includes max_bond_dim, rel_tol, total_tol, renorm, sum_trunc options.
            Defaults to SVDParameters() with max_bond_dim=100.
        
    Returns:
        TreeTensorNetwork: The sum of the two TTNs with controlled bond dimensions
        
    Raises:
        NotCompatibleException: If TTNs have incompatible structures
        ValueError: If TTNs are empty or have mismatched roots
    """
    direct_result = add_ttns_direct_form(ttn1, ttn2)
    
    truncated_result = recursive_truncation(direct_result, svd_params)
    
    return truncated_result

def add_ttns_svd_controlled_params(ttn1: TreeTensorNetwork,ttn2: TreeTensorNetwork,
                          max_bond_dim: int = 100,
                          rel_tol: float = 1e-12,
                          total_tol: float = 1e-12,
                          renorm: bool = False,
                          sum_trunc: bool = False
                          )-> TreeTensorNetwork:
    """
    Wrapper around add_ttns_svd_controlled that allows specifying individual SVD 
    parameters directly without creating an SVDParameters object.
    
    Args:
        ttn1 (TreeTensorNetwork): First TTN to add
        ttn2 (TreeTensorNetwork): Second TTN to add
        max_bond_dim (int, optional): Maximum allowed bond dimension. Defaults to 100.
        rel_tol (float, optional): Relative tolerance for singular values. Defaults to 1e-12.
        total_tol (float, optional): Absolute tolerance for singular values. Defaults to 1e-12.
        renorm (bool, optional): Whether to renormalize after truncation. Defaults to False.
        sum_trunc (bool, optional): Whether to use sum-based truncation. Defaults to False.
    
    Returns:
        TreeTensorNetwork: The sum with controlled bond dimensions
    """
    svd_params = SVDParameters(
        max_bond_dim=max_bond_dim,
        rel_tol=rel_tol,
        total_tol=total_tol,
        renorm=renorm,
        sum_trunc=sum_trunc
    )
    
    return add_ttns_svd_controlled(ttn1, ttn2, svd_params)