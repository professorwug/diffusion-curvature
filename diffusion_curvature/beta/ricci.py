
# %% auto 0
__all__ = ['jax_repeatedly_diffuse_dirac', 'stable_entropy_of_diffusion', 'dirichlet_energy', 'DiffusionRicciCurvature']

# %% ../nbs/library/core-jax/Ricci-Curvature.ipynb 6
import jax.numpy as jnp
def jax_repeatedly_diffuse_dirac(P, i, t):
    signal = jnp.eye(len(P))[i]
    for tt in range(t):
        signal = signal @ P
    return signal

# %% ../nbs/library/core-jax/Ricci-Curvature.ipynb 13
from fastcore.all import *
from typing import Callable, Literal, get_args, get_origin
from .diffusion_laziness import wasserstein_spread_of_diffusion, entropy_of_diffusion
from .heat_diffusion import jax_power_matrix
from tqdm.auto import trange
import jax.numpy as jnp
import jax

_LAZINESS_METHOD = Literal['Entropic', 'Wasserstein', 'Dirichlet Energy']

def stable_entropy_of_diffusion(Pt):
    entropy_elementwise = jax.scipy.special.entr(Pt)
    entropy_of_rows = jnp.sum(entropy_elementwise, axis=-1)
    return entropy_of_rows

def dirichlet_energy(A, F):
    Dnoh = jnp.diag(1/jnp.sqrt(A.sum(axis=1, keepdims=False)))
    L = jnp.eye(len(A)) - (Dnoh @ A @ Dnoh)
    trace = lambda f: jnp.trace(f[:,None].T @ L @ f[:,None])
    return jax.vmap(trace)(F)

class DiffusionRicciCurvature:
    def __init__(self, 
                 laziness_method:_LAZINESS_METHOD = "Entropic", # Method to use for laziness computation
                 allow_self_loops:bool = True, # Whether to allow self-loops in graph. 
                 threshold_eps = 1e-5,
                 ):
        store_attr()
        self.R = None
        match self.laziness_method:
            case "Entropic":
                self.laziness = stable_entropy_of_diffusion
                print("Using stable entropy")
            case "Wasserstein":
                self.laziness = wasserstein_spread_of_diffusion
            case 'Dirichlet Energy':
                self.laziness = dirichlet_energy
            case _:
                raise NotImplementedError(f"No such laziness method {self.laziness_method}")
        
    def fit(
        self,
        A, # Graph affinity matrix
        t:int, # diffusion time
    ):
        self.A = jnp.array(A, dtype=jnp.float32)
        self.A = self.A.at[self.A < self.threshold_eps].set(0)
        if self.allow_self_loops:
            A = A.at[jnp.diag_indices_from(A)].set(0)
        self.A = jax.lax.stop_gradient(self.A)
        self.R = self.ricci_curvature(A, t)
        self.ks = self.scalar_curvature(A, t)
        
    def fit_transform(
        self,
        A, # Graph affinity matrix
        t:int, # diffusion time
    ):
        self.fit(A, t)
        return self.R
        
    def diffusion_laziness_of_graph(self, A, t):
        P = (A / A.sum(axis=1, keepdims=True))
        Pt = jnp.linalg.matrix_power(P, t)
        if self.laziness_method == 'Dirichlet Energy':
            laziness = self.laziness(A, Pt)
        else:
            laziness = -self.laziness(Pt)
        return laziness
    
    def diffusion_laziness_of_idx(self, A, i, t):
        # ALTERNATE IMPLEMENTATION: just use matrix vector products
        # oddly, this is five times slower than powering the whole matrix.
        # Todo: the computation graph is surely being rebuilt repeatedly and unnecessarily every time this is called.
        # P = (A / A.sum(axis=1, keepdims=True))
        # Pt_idx = jax_repeatedly_diffuse_dirac(P, i, t)
        # if self.laziness_method == 'Dirichlet Energy':
        #     laziness = self.laziness(A, Pt_idx[:,None])
        # else:
        #     laziness = -self.laziness(Pt_idx)
        laziness = self.diffusion_laziness_of_graph(A, t)[i]
        return laziness
    
    def diffusion_laziness_of_edge(self, A, i, j, t):
        return self.diffusion_laziness_of_idx(A, i, t) + self.diffusion_laziness_of_idx(A, j, t)
        
    def edge_surgery(self, A, i, j, epsilon):
        # returns A with edges i,j and j,i increased by epsilon
        diager = jnp.zeros_like(A)
        diager = diager.at[[i,j],[j,i]].set(1)
        A = A + epsilon*diager
        return A
    
    def diffusion_laziness_of_edge_with_surgery(self, A, i, j, t, epsilon):
        A_surgery = self.edge_surgery(A, i, j, epsilon)
        # With Diffusion Entropy
        return self.diffusion_laziness_of_edge(A_surgery, i, j, t)
    
    def differential_diffusion_curvature_of_edge(self, A, i, j, t):
        return jax.grad(self.diffusion_laziness_of_edge_with_surgery, argnums=4)(A, i, j, t, 0.0)
    
    def ricci_curvature(self, A, t, eps=1e-8):  
        # return a matrix of the edge wise diffusion curvatures for all edges
        # (i,j) is the edge from i to j
        A = jnp.array(A, dtype=jnp.float32)
        R = jnp.zeros(A.shape)
        edges_used = jnp.zeros(A.shape)
        # get gradient function
        edge_gradient = jax.grad(self.diffusion_laziness_of_edge_with_surgery, argnums=4)
        for i in trange(A.shape[0]):
            for j in range(A.shape[1]):
                if A[i,j] > eps:
                    R = R.at[i,j].set(edge_gradient(A,i,j,t, 0.0))
                    edges_used = edges_used.at[i,j].set(1)
        return R
    
    def scalar_curvature(self, A, t):
        if self.R is None:
            R = self.ricci_curvature(A, t)
        else:
            R = self.R
        ks = jnp.mean(R, axis=-1)
        return ks
        
