
# %% auto 0
__all__ = ['phate_distances_graphtools', 'pairwise_euclidean', 'phate_distances', 'phate_distances_differentiable',
           'phate_distances_from_pointcloud']

# %% ../nbs/library/core-jax/Manifold-Distances.ipynb 3
from sklearn.metrics import pairwise_distances
import numpy as np
import scipy
import graphtools

def phate_distances_graphtools(G:graphtools.api.Graph):
    assert G.Pt is not None
    if type(G.Pt) == np.ndarray:
        log_Pts = -np.log(G.Pt + 1e-6)
        D = pairwise_distances(log_Pts)
    elif type(G.Pt) == scipy.sparse.csr_matrix:
        # TODO: There's likely a more efficient way of doing this. 
        # But I mustn't tempt the devil of premature optimization
        Pt_np = G.Pt.toarray()
        log_Pts = -np.log(Pt_np + 1e-6)
        D = pairwise_distances(log_Pts)
    G.D = D
    return G

# %% ../nbs/library/core-jax/Manifold-Distances.ipynb 4
import jax.numpy as jnp
def pairwise_euclidean(x, y):
  # Pairwise euclidean distances in Jax, courtesy of [jakevdp](https://github.com/google/jax/discussions/11841)
  assert x.ndim == y.ndim == 2
  return jnp.sqrt(((x[:, None, :] - y[None, :, :]) ** 2).sum(-1)) # I would want to use something like PyKeops for this, if being done differentiably.
  
def phate_distances(Pt):
    log_Pts = -jnp.log(Pt + 1e-6)
    D = pairwise_distances(log_Pts, log_Pts)
    return D

def phate_distances_differentiable(Pt):
    return phate_distances(Pt)

# %% ../nbs/library/core-jax/Manifold-Distances.ipynb 5
from .graphs import get_adaptive_graph
from sklearn.preprocessing import normalize
from .heat_diffusion import jax_power_matrix
def phate_distances_from_pointcloud(X, t = 25):
    G = get_adaptive_graph(X)
    P = normalize(G.W, norm="l1", axis=1)
    if type(P) == scipy.sparse._csr.csr_matrix:
        P = P.todense()
    # diffusion_matrix_from_affinities(G.W)
    # P = diff_op(G).todense() # is sparse, by default
    P = jnp.array(P)
    Pt = jax_power_matrix(P,t)
    return phate_distances(Pt)
    
