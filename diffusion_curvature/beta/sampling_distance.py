
# %% auto 0
__all__ = ['sampling_distance']

# %% ../nbs/library/benchmarking/sampling-distance.ipynb 4
from .kernels import gaussian_kernel
import ot
import numpy as np
def sampling_distance(
    X1, # Poincloud 1
    X2, # Pointcloud 2
    dists # Manifold distances for most trusted pointcloud
):
    """
    Returns an EMD between the densities of two distributions.
    Assumes X1 and X2 have pointwise correspondence.
    """
    
    # A fixed kernel is best here. We estimate sigma based on the distances between neighbors, ensuring consistency between datasets.
    A1 = gaussian_kernel(X1,kernel_type="fixed",sigma=0, anisotropic_density_normalization = 0)
    D1 = A1.sum(1) # Get the degree as a density proxy
    D1 /= D1.sum() # normalize to one, as a distribution

    # repeat for X2
    A2 = gaussian_kernel(X2,kernel_type="fixed",sigma=0, anisotropic_density_normalization = 0)
    D2 = A2.sum(1) # Get the degree as a density proxy
    D2 /= D2.sum() # normalize to one, as a distribution

    return ot.emd2(D1, D2, dists)
