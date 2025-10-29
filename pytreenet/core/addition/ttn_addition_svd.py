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
    Adds two Tree Tensor Networks (TTN) using direct form addition followed by SVD truncation
    to manage bond dimension growth.

    Args:
        ttn1 (TreeTensorNetwork): First TTN
        ttn2 (TreeTensorNetwork): Second TTN
        svd_params: SVD parameters for truncation. Defaults to max_bond_dim=100.
        
    Returns:
        TreeTensorNetwork: Sum of the two TTNs
    Raises:
        NotCompatibleException: Incompatible TTN structures
        ValueError: Empty TTNs or mismatched roots
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
    Adds two TTNs with SVD truncation, accepting individual parameters instead of 
    SVDParameters object.
    
    Args:
        ttn1 (TreeTensorNetwork): First TTN
        ttn2 (TreeTensorNetwork): Second TTN
        max_bond_dim (int, optional): Maximum bond dimension. Defaults to 100.
        rel_tol (float, optional): Relative tolerance for singular values. Defaults to 1e-12.
        total_tol (float, optional): Absolute tolerance for singular values. Defaults to 1e-12.
        renorm (bool, optional): Renormalize after truncation. Defaults to False.
        sum_trunc (bool, optional): Use sum-based truncation. Defaults to False.
    
    Returns:
        TreeTensorNetwork: Sum with controlled bond dimensions
    """
    svd_params = SVDParameters(
        max_bond_dim=max_bond_dim,
        rel_tol=rel_tol,
        total_tol=total_tol,
        renorm=renorm,
        sum_trunc=sum_trunc
    )
    
    return add_ttns_svd_controlled(ttn1, ttn2, svd_params)