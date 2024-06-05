__all__ = ['PointCloudFlattener', 'EuclideanComparisonSpace', 'fit_comparison_space_model', 'get_graph_type',
           'euclidean_comparison_graph', 'construct_ndgrid', 'construct_ndgrid_from_shape', 'diffusion_coordinates',
           'load_average_entropies']

import pygsp
import jax
import jax.numpy as jnp
import numpy as np
from fastcore.all import *
import skdim
import scipy
from sklearn.preprocessing import normalize

from inspect import getfullargspec
from typing import Callable, Literal, get_args, get_origin
import graphtools
from tqdm.auto import trange, tqdm

from jax.experimental import sparse

# Graph operations, laziness, measures
from .graphs import diff_aff, diff_op, diffusion_matrix_from_affinities
from .heat_diffusion import heat_diffusion_on_signal, kronecker_delta, jax_power_matrix, heat_diffusion_from_dirac
from .diffusion_laziness import wasserstein_spread_of_diffusion, entropy_of_diffusion
from .distances import phate_distances
from .datasets import plane

# Comparison space construction
# from diffusion_curvature.normalizing_flows import neural_flattener
# from diffusion_curvature.flattening.mioflow_quicktrain import MIOFlowStandard
from .flattening.radial_ae import radially_flatten_with_ae
from .local_distance_scaling import scale_by_local_distances

# Algorithmic niceties
from .clustering import enhanced_spectral_clustering
from .vne import optimal_t_via_vne
from .utils import random_jnparray

import diffusion_curvature

class PointCloudFlattener:
    def __init__(self, 
                 flattening_method,
                 graph_former, 
                 diffusion_curvature_instance, 
                 comparison_space_file = "../data/entropies_averaged.h5",
                 verbose=False):
        self.flattening_method = flattening_method
        self.graph_former = graph_former
        self.DC = diffusion_curvature_instance
        self.verbose = self.DC.verbose
        if flattening_method == "Mean Fixed":
            self.SGT = load_average_entropies(comparison_space_file)
        
    def fit_transform(self, X, dimension, graph_former, num_points_in_comparison = None):
        return self.get_flat_graph(X, dimension, graph_former, num_points_in_comparison)
    
    def get_flat_graph(self, X, dimension, graph_former, num_points_in_comparison = None):
        if num_points_in_comparison is None:
            num_points_in_comparison = X.shape[0]
        
        match self.flattening_method:
            case "Fixed":
                # if self.verbose: print(f"{num_points_in_comparison=}")
                Rn = plane(n = int(num_points_in_comparison), dim=dimension)
                G_euclidean = graph_former(Rn)
                
            # case "Mean Fixed":
            #     dimension_checks_out = dimension in self.SGT.keys()
            #     knn_checks_out = knn in self.SGT[dimension].keys() if dimension_checks_out else False
            #     t_checks_out = t in self.SGT[dimension][knn].keys() if knn_checks_out else False
            #     if not (dimension_checks_out and knn_checks_out and t_checks_out):
            #         # compute the old way
            #         print("Flat space not precomputed; computing now")
            #         self.flattening_method = "Fixed"
            #         return get_flat_spreads(
            #             dimension = dimension,
            #             jump_of_diffusion = jump_of_diffusion,
            #             num_points_in_comparison = num_points_in_comparison,
            #             cluster_idxs = cluster_idxs,
            #             verbose=verbose
            #             )
            #     else:
            #         return self.SGT[dimension][knn][t]  
                
            case "Scaled Fixed":
                Rd = plane(n = num_points_in_comparison, dim=dimension)
                Rd = scale_by_local_distances(X, Rd, k = 1)
                G_euclidean = self.graph_former(Rd)
        return G_euclidean

# %% ../nbs/library/core-jax/Comparison-Space-Construction.ipynb 9
from .graphs import generic_kernel, diffusion_matrix_from_affinities
from .distances import phate_distances_differentiable, pairwise_euclidean
from .utils import random_jnparray
from .diffusion_laziness import wasserstein_spread_of_diffusion, entropy_of_diffusion
import jax
import jax.numpy as jnp
from flax import linen as nn
import optax

class EuclideanComparisonSpace(nn.Module):
    dimension:int  # dimension of comparison space
    num_points:int # num points to sample; best determined as a subset of the number of points in your manifold.
    jump_of_diffusion:jax.Array # the W1 distance from a single step of diffusion (t=1) to its origin
    fraction_of_points:float = 0.8
    comparison_type:str = "entropy"

    def setup(self):
        # compute optimal number of points given dimension
        self.Rn = jnp.concatenate([jnp.zeros((1,self.dimension)), 2*random_jnparray(self.num_points-1, self.dimension)-1])
        self.num_useful_points = int(self.fraction_of_points * self.num_points)
        # Sort values by distance to center point (i.e. the origin)
        distances_to_center = (self.Rn**2).sum(-1)
        sorting_idxs = jnp.argsort(distances_to_center)
        self.Rn = self.Rn[sorting_idxs]
        # precompute distances for kernel
        self.D = pairwise_euclidean(self.Rn,self.Rn)
        # initialize tunable parameters
        sigma_base = nn.initializers.constant(0.7)
        self.sigma = self.param(
            'kernel bandwidth',
            sigma_base, # Initial value of kernel bandwidth
            1 # size - it's just one value.
        )
        # anisotropy_base = nn.initializers.constant(0.5)
        self.anisotropic_density_normalization = 1
        # self.anisotropic_density_normalization = self.param(
        #     'anisotropic normalization',
        #     anisotropy_base, # Initial value of kernel bandwidth
        #     1 # size - it's just one value.
        # )


    def __call__(self):
        # normalize anisotropic density normalization to fit between 0 and 1
        # print(f"sigma = {self.sigma} alpha = {self.anisotropic_density_normalization}")
        A = generic_kernel(self.D, self.sigma, jax.nn.relu(self.anisotropic_density_normalization))
        P = diffusion_matrix_from_affinities(A)
        # print(f"AFTER ALL THAT - the diffusion matrix has max {jnp.max(P)}")
        if jnp.min(P) < 0: raise ValueError("P has negative values ", jnp.min(P))
        # D_manifold = phate_distances_differentiable(P)
        match self.comparison_type:
            case "wasserstein":
                W1 = wasserstein_spread_of_diffusion(self.D,P) # vector of all W1 values in comparison space.
                # discard edge values; take only top 80% of W1s, corresponding to closest 80% of values.
                # TODO: 80% is arbitrary and could be improved.
                spreads_near_center = W1[:-self.num_useful_points]
            case "entropy":
                # print(P)
                H = entropy_of_diffusion(P)
                spreads_near_center = H[:-self.num_useful_points]
        
        return {
            'mean jump difference':jnp.abs(jnp.mean(spreads_near_center) - self.jump_of_diffusion), 
            'A':A, 
            'P':P, 
            'D':self.D
        }

# %% ../nbs/library/core-jax/Comparison-Space-Construction.ipynb 11
import optax
import jax.random as random
def fit_comparison_space_model(model, max_epochs = 1000, verbose=False, lr = 1e-3):
    def train_comparison_space_model(params):
        out_dict = model.apply(params)
        return out_dict['mean jump difference']
    key1, key2 = jax.random.split(jax.random.key(0))
    params = model.init(key2)
    tx = optax.adam(learning_rate=lr)
    opt_state = tx.init(params)
    loss_grad_fn = jax.value_and_grad(train_comparison_space_model)
    for i in range(max_epochs):
        loss_val, grads = loss_grad_fn(params)
        if loss_val < 1e-5: break
        updates, opt_state = tx.update(grads,opt_state)
        params = optax.apply_updates(params,updates)
        if verbose:
            if i % 10 == 0: print(f"at {i}, loss is {loss_val}")
        if loss_val != loss_val: raise ValueError("NANS! Look out!")
    if verbose: print("ending with loss value", loss_val)
    return params

# %% ../nbs/library/core-jax/Comparison-Space-Construction.ipynb 25
import pygsp
import numpy as np
import graphtools

def get_graph_type(G):
    tt = str(type(G)).split('.')[-1][:3].lower()
    # TODO: Support for MNN graphs
    if tt == 'mnn': raise NotImplementedError("MNN Graphs require manual indices. These are not yet supported by our graph flattener.")
    if tt in ['knn']: return tt
    else: return 'exact'

def euclidean_comparison_graph(G:pygsp.graphs.Graph, dimension):
        """
        Constructs a flat graph, hewn from uniform random noise of the supplied dimension.
        Calculates the powered diffusion matrix on this graph.
        """
        noise = np.concatenate([np.zeros((1,dimension)), 2*np.random.rand(G.K.shape[0]-1,dimension)-1])
        # Build a graph out of the noise, with all of the same kernel settings as our first graph
        # NOTE: The graph building settings must be scale invariant! 
        params = G.get_params()
        needed_keys = ['data', 'n_pca', 'knn', 'decay', 'bandwidth', 'bandwidth_scale', 
                    'knn_max', 'anisotropy', 'beta', 'adaptive_k', 'n_landmark', 'n_landmark', 
                    'n_svd', 'n_jobs']
        found_keys = {} # TODO: Likely a more elegant way to do this
        for nk in needed_keys:
            if nk in params.keys():
                found_keys[nk] = params[nk]
            else:
                found_keys[nk] = None
        
        G_flat = graphtools.Graph(
            data = noise,
            n_pca = found_keys['n_pca'],
            knn = found_keys['knn'],
            decay = found_keys['decay'],
            bandwidth = found_keys['bandwidth'],
            bandwidth_scale = found_keys['bandwidth_scale'],
            knn_max = found_keys['knn_max'],
            anisotropy = found_keys['anisotropy'],
            beta = found_keys['beta'],
            adaptive_k = found_keys['adaptive_k'],
            n_landmark = found_keys['n_landmark'],
            n_svd = found_keys['n_svd'],
            n_jobs = found_keys['n_jobs'],
            graphtype = get_graph_type(G),
        )
        return G_flat.to_pygsp()

# %% ../nbs/library/core-jax/Comparison-Space-Construction.ipynb 35
import numpy as np
def construct_ndgrid(*args):
    # Construct an ndgrid of points
    ndgrid = np.meshgrid(*args, indexing='ij')
    points = np.vstack(list(map(np.ravel, ndgrid))).T
    return points
def construct_ndgrid_from_shape(dim, points_per_dim):
    # Construct an ndgrid of points
    ranges = [np.arange(start=-1,stop=1,step=2/points_per_dim) for _ in range(dim)]
    points = construct_ndgrid(*ranges)
    # move the element closest to the origin to the front
    distances_to_origin = (points**2).sum(-1)
    sorting_idxs = np.argsort(distances_to_origin, )
    points = points[sorting_idxs]
    return points

# %% ../nbs/library/core-jax/Comparison-Space-Construction.ipynb 40
import jax.numpy as jnp
from .graphs import diff_aff

def diffusion_coordinates(G, t = 1, plot_evals = False):
    P_symmetric = jnp.array(diff_aff(G).toarray())
    W = jnp.array(G.W.toarray())
    D = jnp.sum(W, axis=1)
    # given symmetric diffusion matrix and density, constructs diffusion map
    Dnoh = jnp.diag(D**-0.5)
    # Decompose Ms
    eig_vals, eig_vecs = jnp.linalg.eigh(P_symmetric)
    # sort eigenvalues and eigenvectors(they are inconsistently sorted by default)
    sorted_idxs = jnp.argsort(eig_vals)
    eig_vals = eig_vals[sorted_idxs]
    eig_vecs = eig_vecs[:,sorted_idxs]
    # Normalize the eigenvector
    eig_psi_components = Dnoh @ eig_vecs
    eig_psi_components = eig_psi_components @ jnp.diag(jnp.power(jnp.linalg.norm(eig_psi_components, axis=0), -1))
    # Remove the trivial eigenvalue and eigenvector
    eig_vals = eig_vals[:-1]
    if plot_evals:
        print(eig_vals)
        fig, ax = plt.subplots()
        ax.bar([str(i) for i in range(len(eig_vals))], eig_vals**t)
        ax.set_title("Evals")
        plt.show()
    eig_psi_components = eig_psi_components[:,:-1]
    # Construct the diffusion map
    # diff_map = eig_psi_components @ np.diag(eig_vals**t)
    diff_map = eig_vals**t * eig_psi_components
    diff_map = diff_map[:,::-1]
    diff_map = diff_map
    return diff_map


import h5py
def load_average_entropies(filename):
    d = {}
    with h5py.File(filename,'r') as f:
        for dim in f.keys():
            d[int(dim)] = {}
            for knn in f[dim].keys():
                d[int(dim)][int(knn)] = {}
                for t in f[dim][knn].keys():
                    d[int(dim)][int(knn)][int(t)] = f[dim][knn][t][()]
    return d
