# %% auto 0
__all__ = ['median_heuristic', 'gaussian_kernel', 'compute_anisotropic_affinities_from_graph',
           'compute_anisotropic_diffusion_matrix_from_graph', 'pygsp_graph_from_points', 'get_curvature_agnostic_graph',
           'get_adaptive_graph', 'get_fixed_graph', 'get_knn_graph', 'diffusion_matrix']

# %% ../nbs/library/core-jax/Kernels.ipynb 7
import numpy as np
def median_heuristic(
        D:np.ndarray, # the distance matrix
):
    # estimate kernel bandwidth from distance matrix using the median heuristic
    # Get upper triangle from distance matrix (ignoring duplicates)
    h = D[np.triu_indices_from(D)]
    h = h**2
    h = np.median(h)
    nu = np.sqrt(h / 2)
    return nu

# %% ../nbs/library/core-jax/Kernels.ipynb 8
import numpy as np
from sklearn.metrics import pairwise_distances
def gaussian_kernel(
        X:np.ndarray, # pointcloud data as rows, shape n x d
        kernel_type = "fixed", # either fixed, or adaptive
        sigma:float = 0, # if fixed, uses kernel bandwidth sigma. If not set, uses a heuristic to estimate a good sigma value
        k:float = 10, # if adaptive, creates a different kernel bandwidth for each point, based on the distance from that point to the kth nearest neighbor
        anisotropic_density_normalization:float = 0.5, # if nonzero, performs anisotropic density normalization
        threshold_for_small_values:float = 1e-5, # Sets all affinities below this value to zero. Set to zero to disable.
        neighbor_scale:float = 3.0, # if curvature agnostic, this is the scale of the neighbor distance
):
    """Constructs an affinity matrix from pointcloud data, using a gaussian kernel"""
    supported_kernel_types = {'fixed', 'adaptive', 'curvature agnostic'}
    assert kernel_type in supported_kernel_types
    D = pairwise_distances(X)
    if kernel_type == "fixed":
        if not sigma:
            # estimate sigma using a heuristic
            sigma = median_heuristic(D)
        W = (1/(sigma*np.sqrt(2*np.pi)))*np.exp((-D**2)/(2*sigma**2))
    elif kernel_type == "adaptive":
        distance_to_k_neighbor = np.partition(D,k)[:,k]
        # Populate matrices with this distance for easy division.
        div1 = np.ones(len(D))[:,None] @ distance_to_k_neighbor[None,:]
        div2 = distance_to_k_neighbor[:,None] @ np.ones(len(D))[None,:]
        # print("Distance to kth neighbors",distance_to_k_neighbor)
        # compute the gaussian kernel with an adaptive bandwidth
        W = (1/(2*np.sqrt(2*np.pi)))*(np.exp(-D**2/(2*div1**2))/div1 + np.exp(-D**2/(2*div2**2))/div2)
    elif kernel_type == "curvature agnostic":
        scaled_neighbor_dists = np.partition(D,k)[:,k]  # TODO is multiplication best here?
        sigma = np.mean(scaled_neighbor_dists) * neighbor_scale
        W = np.exp((-D**2)/sigma**2)
        # div1 = np.ones(len(D))[:,None] @ scaled_neighbor_dists[None,:]
        # div2 = scaled_neighbor_dists[:,None] @ np.ones(len(D))[None,:]
        # W = (1/(2*np.sqrt(2*np.pi)))*(np.exp(-D**2/(2*div1**2))/div1 + np.exp(-D**2/(2*div2**2))/div2)
    if anisotropic_density_normalization:
        D = np.diag(1/(np.sum(W,axis=1)**anisotropic_density_normalization))
        W = D @ W @ D
    if threshold_for_small_values:
        W[W < threshold_for_small_values] = 0
    return W

# %% ../nbs/library/core-jax/Kernels.ipynb 9
import numpy as np
def compute_anisotropic_affinities_from_graph(
    W:np.ndarray, # the adjacency/affinity matrix of the graph
    alpha:float, # the anisotropic density normalization parameter
) -> np.ndarray:
    # normalize by density
    D = np.diag(1/(np.sum(W,axis=1)**alpha))
    W = D @ W @ D
    return W

def compute_anisotropic_diffusion_matrix_from_graph(
    A:np.ndarray, # the adjacency/affinity matrix of the graph
    alpha:float, # the anisotropic density normalization parameter
    ) -> np.ndarray:
    A_anis = compute_anisotropic_affinities_from_graph(A,alpha)
    # row normalize to create diffusion matrix
    D = np.diag(1/(np.sum(A_anis,axis=1)+1e-8))
    P = D @ A_anis
    return P

# %% ../nbs/library/core-jax/Kernels.ipynb 10
import pygsp
def pygsp_graph_from_points(X, knn=15):
    W = gaussian_kernel(X, kernel_type="adaptive", k=knn, anisotropic_density_normalization=1)
    G = pygsp.graphs.Graph(W)
    return G

# %% ../nbs/library/core-jax/Kernels.ipynb 11
def get_curvature_agnostic_graph(X, neighbor_scale = 1, k = 1, alpha = 1):
    W = gaussian_kernel(
        X, 
        kernel_type = "curvature agnostic", 
        k = k, 
        neighbor_scale=neighbor_scale, 
        anisotropic_density_normalization = alpha)
    # set diagonal of W to zero
    np.fill_diagonal(W, 0)
    G = pygsp.graphs.Graph(W)
    return G

def get_adaptive_graph(X, k = 5, alpha = 1):
    W = gaussian_kernel(
        X,
        kernel_type='adaptive',
        k = k,
        anisotropic_density_normalization = alpha,
    )
    G = pygsp.graphs.Graph(W)
    return G

def get_fixed_graph(X, sigma = 0.2, alpha = 1):
    W = gaussian_kernel(
        X, 
        kernel_type = "fixed",
        sigma = sigma,
        anisotropic_density_normalization = alpha,
    )
    # set diagonal of W to zero
    np.fill_diagonal(W, 0)
    G = pygsp.graphs.Graph(W)
    return G

# %% ../nbs/library/core-jax/Kernels.ipynb 13
from sklearn.neighbors import kneighbors_graph
def get_knn_graph(
        X:np.ndarray,
        k = 10,
        alpha = 1,
        use_pygsp = True,
        self_loops = True
):
    W = kneighbors_graph(X, k, mode='connectivity', include_self=self_loops).toarray()
    if alpha > 0:
        W = compute_anisotropic_affinities_from_graph(W, alpha)
    if use_pygsp:
        G = pygsp.graphs.Graph(W)
        return G
    else:
        return W

# %% ../nbs/library/core-jax/Kernels.ipynb 16
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize
from graphtools.matrix import set_diagonal

def diffusion_matrix(
        X:np.ndarray = None, # pointcloud data
        A:np.ndarray = None, # adjacency matrix, if precomputed
        kernel_type:str = "fixed", # either fixed or adaptive
        sigma = 0, # if fixed, uses kernel bandwidth sigma. If not set, uses a heuristic to estimate a good sigma value
        k = 10, # if adaptive, creates a different kernel bandwidth for each point, based on the distance from that point to the kth nearest neighbor
        anisotropic_density_normalization = 0.5, # if nonzero, performs anisotropic density normalization
        threshold_for_small_values = 1e-5,
):
    """ Creates a diffusion matrix from pointcloud data, by row-normalizing the affinity matrix obtained from the gaussian_kernel function """
    if X is not None:
        W = gaussian_kernel(X,kernel_type,sigma=sigma,k = k,anisotropic_density_normalization = anisotropic_density_normalization, threshold_for_small_values=threshold_for_small_values)
        W = W + np.eye(len(X))*1e-5
    if X is None and A is not None:
        W = A
    
    K = set_diagonal(W, 1)
    P = normalize(W, norm="l1", axis=1)
    return P
