# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/library/core-jax/Core.ipynb.

# %% auto 0
__all__ = ['default_fixed_graph_former', 'graphtools_graph_from_data', 'SimpleGraph', 'get_adaptive_graph', 'fixed_graph_former',
           'DiffusionCurvature', 'fill_diagonal']

# %% ../../nbs/library/core-jax/Core.ipynb 10
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
from .comparison_space import EuclideanComparisonSpace, fit_comparison_space_model, euclidean_comparison_graph, construct_ndgrid_from_shape, diffusion_coordinates, load_average_entropies
from .normalizing_flows import neural_flattener
from .flattening.mioflow_quicktrain import MIOFlowStandard
from .flattening.radial_ae import radially_flatten_with_ae

# Algorithmic niceties
from .clustering import enhanced_spectral_clustering
from .vne import optimal_t_via_vne
from .utils import random_jnparray

import diffusion_curvature

import torch

# import deepdish
import h5py

def graphtools_graph_from_data(X):
    return graphtools.Graph(X, anisotropy=1, knn=15, decay=None).to_pygsp()

from .kernels import gaussian_kernel
from dataclasses import dataclass

@dataclass
class SimpleGraph:
    W: np.ndarray

def get_adaptive_graph(X, k = 5, alpha = 1):
    W = gaussian_kernel(
        X,
        kernel_type='adaptive',
        k = k,
        anisotropic_density_normalization = alpha,
    )
    G = pygsp.graphs.Graph(W)
    return G

def fixed_graph_former(X, sigma, alpha):
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

default_fixed_graph_former = partial(fixed_graph_former, sigma = 0.2, alpha = 1)

_DIFFUSION_TYPES = Literal['diffusion matrix','heat kernel']
_LAZINESS_METHOD = Literal['Wasserstein','Entropic', 'Laziness', 'Wasserstein Normalized']
_FLATTENING_METHOD = Literal['Neural', 'Fixed', 'Mean Fixed', 'MIOFlow', 'Radial Flattener']
_COMPARISON_METHOD = Literal['Ollivier', 'Subtraction']

class DiffusionCurvature():
    def __init__(
            self,
            diffusion_type:_DIFFUSION_TYPES = 'diffusion matrix', # Either ['diffusion matrix','heat kernel']
            laziness_method: _LAZINESS_METHOD = 'Wasserstein', # Either ['Wasserstein','Entropic', 'Laziness']
            flattening_method: _FLATTENING_METHOD = 'Fixed', # Either ['Neural', 'Fixed', 'Mean Fixed', 'MIOFlow']
            comparison_method: _COMPARISON_METHOD = 'Subtraction', # Either ['Ollivier', 'Subtraction']
            graph_former = default_fixed_graph_former,
            dimest = None, # Dimension estimator to use. If none, defaults to kNN.
            points_per_cluster = None, # Number of points to use in each cluster when constructing comparison spaces. Each comparison space takes about 20sec to construct, and has different sampling and dimension. If 1, constructs a different comparison space for each point; if None, constructs just one comparison space.
            comparison_space_size_factor = 1, # Number of points in comparison space is the number of points in the original space divided by this factor.
            use_grid=False, # If True, uses a grid of points as the comparison space. If False, uses a random sample of points.            
            max_flattening_epochs=50,     
            aperture = 20, # if using Laziness flattening, this controls the size of neighborhood over which the return probability is averaged.
            smoothing=1,
            distance_t = None,
            comparison_space_file = "../data/entropies_averaged.h5",
            verbose = False,
    ):
        store_attr()
        self.D = None
        self.laziness = None
        self.graph_former = graph_former
        if self.dimest is None:
            self.dimest = skdim.id.KNN()
        if self.flattening_method == "Mean Fixed":
            self.SGT = load_average_entropies(comparison_space_file)
            # deepdish.io.load("../data/sgt_peppers_averaged_flat_entropies.h5") # dict of dim x knn x ts containing precomputed flat entropies.

        
    def unsigned_curvature(
            self,
            G:pygsp.graphs.Graph, # PyGSP input Graph
            t:int, # Scale at which to compute curvature; number of steps of diffusion.
            idx=None, # the index at which to compute curvature. If None, computes for all points. TODO: Implement
            # The below are used internally
            _also_return_first_scale = False, # if True, calculates the laziness measure at both specified t and t=1. The distances (if used) are calcualted with the larger t.
            D = None, # Supply manifold distances yourself to override their computation. Only used with the Wasserstein laziness method.
            
    ):
        n = G.W.shape[0]
        # Compute diffusion matrix
        match self.diffusion_type:
            case 'diffusion matrix':
                # if W has no self loops, i.e. zeros on the diagonal, then we must add ones to the diagonal
                # if np.all(G.W.diagonal() == 0):
                #     G.W.setdiag(1)
                P = normalize(G.W, norm="l1", axis=1)
                if type(P) == scipy.sparse._csr.csr_matrix:
                    P = P.todense()
                # diffusion_matrix_from_affinities(G.W)
                # P = diff_op(G).todense() # is sparse, by default
                self.P = jnp.array(P)
                if t is None: t = optimal_t_via_vne(P)
                self.Pt = jax_power_matrix(self.P,t)
                if self.distance_t is not None:
                    self.P_dist = jax_power_matrix(self.P,self.distance_t)
                else:
                    self.P_dist = jax_power_matrix(self.P, t)
            case 'heat kernel':
                if self.distance_t is None:
                    self.distance_t = t
                if t is None: 
                    normal_P = normalize(G.W, norm="l1", axis=1)
                    if type(normal_P) == scipy.sparse._csr.csr_matrix:
                        normal_P = normal_P.todense()
                    normal_P = jnp.array(normal_P)
                    t = optimal_t_via_vne(normal_P)
                Ps = heat_diffusion_from_dirac(G, idx=idx, t=[1,self.distance_t, t])
                # signal = jnp.eye(n) if idx is not None else kronecker_delta(n,idx=idx)
                # Ps = heat_diffusion_on_signal(G, signal, [1,t])
                self.P = Ps[0]
                self.P_dist = Ps[1]
                self.Pt = Ps[2]
            case _:
                raise ValueError(f"Diffusion Type {self.diffusion_type} not in {_DIFFUSION_TYPES}")
        match self.laziness_method:
            case "Wasserstein":
                if D is None: 
                    # raise NotImplementedError("If using Wasserstein-style diffusion curvature, you must pass in precomputed manifold distances with the 'D = ' parameter. If you don't want to compute those, we recommend setting the laziness type to 'Entropic'")
                    D = phate_distances(self.P_dist) #TODO: Could be more efficient here if there's an idx
                self.D = D
                laziness = wasserstein_spread_of_diffusion(D,self.Pt) if idx is None else wasserstein_spread_of_diffusion(D[idx],self.Pt[idx])
                if _also_return_first_scale: laziness_nought = wasserstein_spread_of_diffusion(D,self.P)
            case "Wasserstein Normalized":
                if D is None: 
                    if self.distance_t is not None:
                        self.P_dist = jax_power_matrix(self.P,self.distance_t)
                    else:
                        self.P_dist = jax_power_matrix(self.P, t)
                    # raise NotImplementedError("If using Wasserstein-style diffusion curvature, you must pass in precomputed manifold distances with the 'D = ' parameter. If you don't want to compute those, we recommend setting the laziness type to 'Entropic'")
                    D = phate_distances(self.P_dist) #TODO: Could be more efficient here if there's an idx
                laziness = wasserstein_spread_of_diffusion(D,self.Pt) if idx is None else wasserstein_spread_of_diffusion(D[idx],self.Pt[idx])
                laziness_nought = wasserstein_spread_of_diffusion(D,self.P) if idx is None else wasserstein_spread_of_diffusion(D[idx],self.P[idx])
                laziness = laziness / (laziness_nought @ jax_power_matrix(self.P, self.smoothing))
            case "Entropic":
                laziness = entropy_of_diffusion(self.Pt) if idx is None else entropy_of_diffusion(self.Pt[idx])
                # laziness = entropy_of_diffusion(self.P) / entropy_of_diffusion(self.Pt)
                if _also_return_first_scale: laziness_nought = entropy_of_diffusion(self.P)
            case "Laziness":
                thresholds = jnp.partition(self.P,-self.aperture)[:,-self.aperture] # aperture controls the size of the neighborhood in which laziness is measured
                P_thresholded = (self.P >= thresholds[:,None]).astype(int) 
                near_neighbors_only = self.Pt * P_thresholded
                laziness_aggregate = jnp.sum(near_neighbors_only,axis=1)
                # divide by the number of neighbors diffused to
                ones_remaining = jnp.ones_like(P_thresholded) * P_thresholded # is this needed? Isn't ones_remaining identical to P_thresholded?
                local_density = jnp.sum(ones_remaining,axis=1)
                if self.verbose: print("local density",local_density)
                local_density = local_density.at[local_density==0].set(1) # In case of isolated points, replace local density of 0 with 1. THe laziness will evaluate to zero.
                laziness_aggregate = laziness_aggregate / local_density 
                laziness = laziness_aggregate
                if self.smoothing: # TODO there are probably more intelligent ways to do this smoothing
                    # Local averaging to counter the effects local density
                    if self.verbose: print("Applying smoothing...")
                    smoothing_P_powered = jnp.linalg.matrix_power(self.P,self.smoothing)
                    average_laziness = smoothing_P_powered @ laziness_aggregate[:,None]
                    laziness = average_laziness.squeeze()
                    
                if _also_return_first_scale: laziness_nought = jnp.sum(self.P * P_thresholded,axis=1)
            case _:
                raise ValueError(f"Laziness Method {self.laziness_method} not in {_LAZINESS_METHOD}")
        self.laziness = laziness
        if _also_return_first_scale: 
            return laziness, laziness_nought, self.P, self.Pt, t
        else:
            return laziness
            
    def curvature(
            self,
            G:pygsp.graphs.Graph, # Input Graph
            t:int, # Scale; if none, finds the knee-point of the spectral entropy curve of the diffusion operator
            idx=None, # the index at which to compute curvature. If None, computes for all points.
            dim = None, # the INTRINSIC dimension of your manifold, as an int for global dimension or list of pointwise dimensions; if none, tries to estimate pointwise.
            knn = 15, # Number of neighbors used in construction of graph;
            D = None, # Supply manifold distances yourself to override their computation. Only used with the Wasserstein laziness method.
            X = None, # if using a flattening method that requires a point cloud, supply it here.
    ):
        fixed_comparison_cache = {} # if using a fixed comparison space, saves by dimension
        def get_flat_spreads(dimension, jump_of_diffusion, num_points_in_comparison, cluster_idxs, verbose=False):
            match self.flattening_method:
                case "Fixed":
                    if self.verbose: print(f"{num_points_in_comparison=}")
                    Rn = plane(n = int(num_points_in_comparison), dim=dimension)
                    G_euclidean = self.graph_former(Rn)
                    # if dimension not in fixed_comparison_cache.keys():
                    #     if self.use_grid:
                    #         Rn = construct_ndgrid_from_shape(dimension, int(num_points_in_comparison**(1/dimension)))
                    #     else:
                    #         Rn = jnp.concatenate([jnp.zeros((1,dim)), 2*random_jnparray(num_points_in_comparison-1, dim)-1])
                    #     # construct a lattice in dim dimensions of num_points_in_comparison points
                    #     G = self.graph_former(Rn)
                    #     # G = graphtools.Graph(Rn, anisotropy=1, knn=knn, decay=None,).to_pygsp()
                    #     if self.laziness_method == "Wasserstein": 
                    #         fixed_comparison_cache[dimension] = (G, scipy.spatial.distance_matrix(Rn,Rn))
                    #     else: 
                    #         fixed_comparison_cache[dimension] = (G, None)
                    # G_euclidean, D_euclidean = fixed_comparison_cache[dimension]
                    # print(type(G_euclidean))
                    # print(G_euclidean.W)
                    # scale the euclidean distances to preserve the maximum radial distance from the center
                    # if D is not None:
                    #     # find idx of closest points; get distances between these
                    #     cluster_dists_to_center = D[cluster_idxs][:,cluster_idxs][0]
                    #     print(cluster_dists_to_center.shape)
                    #     manifold_neighb_dists = jnp.sort(cluster_dists_to_center)[:knn]
                    #     euclidean_neighb_dists = jnp.sort(D_euclidean[0])[:knn]
                    #     # scale average euclidean dist to match average manifold dist to closest neighborhoods
                    #     # TODO: this assumes that the distances increase linearly... Not true for, e.g. diffusion distances.
                    #     scaling_factor = jnp.mean(manifold_neighb_dists)/jnp.mean(euclidean_neighb_dists)
                    #     # scaling_factor = jnp.max(D[cluster_idxs][0])/jnp.max(D_euclidean[0])
                    #     D_euclidean = D_euclidean * scaling_factor
                    fs = self.unsigned_curvature(G_euclidean,t,idx=0) #,D=D_euclidean)   
                    if self.verbose: print(f"comparison entropy is {fs}")
                    return fs
                case "Kernel Matching":
                    model = EuclideanComparisonSpace(dimension=dimension, num_points=num_points_in_comparison, jump_of_diffusion=jump_of_diffusion,)
                    params = fit_comparison_space_model(model, max_epochs=1000)
                    if verbose: print(params)
                    euclidean_stuffs = model.apply(params) # dictionary containing A, P, D
                    # W = fill_diagonal(euclidean_stuffs['A'],0)
                    G_euclidean = SimpleGraph(euclidean_stuffs['A'])
                    fs = self.unsigned_curvature(G_euclidean,t,idx=0)
                    return fs
                case "Neural":
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    NF = neural_flattener(device=device, max_epochs=self.max_flattening_epochs)
                    # for now, we assume that the neural flattener is only used with single point clusters
                    # TODO: generalize to multiple point clusters by finding centroid
                    distances_to_manfred = jnp.sum(jnp.array(
                        [jnp.linalg.norm(self.diff_coords - self.diff_coords[clustidx],axis=1) for clustidx in cluster_idxs]
                    ),axis=1)
                    idx_closest_to_manfred = jnp.argsort(distances_to_manfred)[:num_points_in_comparison]
                    diff_coords_of_comparison_space = self.diff_coords[idx_closest_to_manfred]
                    flattened_diff_coords = NF.fit_transform(
                        torch.tensor(diff_coords_of_comparison_space.tolist())
                    )
                    # construct graph out of these flattened coordinates
                    G_euclidean = self.graph_former(flattened_diff_coords)
                    # graphtools.Graph(flattened_diff_coords, knn=15, decay=None, anisotropy=1).to_pygsp()
                    fs = self.unsigned_curvature(G_euclidean, t, idx=0)
                    return fs
                    # return G_euclidean, None # TODO: compute diffusion distances
                case "Mean Fixed":
                    dimension_checks_out = dimension in self.SGT.keys()
                    knn_checks_out = knn in self.SGT[dimension].keys() if dimension_checks_out else False
                    t_checks_out = t in self.SGT[dimension][knn].keys() if knn_checks_out else False
                    if not (dimension_checks_out and knn_checks_out and t_checks_out):
                        # compute the old way
                        print("Flat space not precomputed; computing now")
                        self.flattening_method = "Fixed"
                        return get_flat_spreads(
                            dimension = dimension,
                            jump_of_diffusion = jump_of_diffusion,
                            num_points_in_comparison = num_points_in_comparison,
                            cluster_idxs = cluster_idxs,
                            verbose=verbose
                            )
                    else:
                        return self.SGT[dimension][knn][t]    
                case "MIOFlow":
                    self.NeuralFlattener = MIOFlowStandard(
                            embedding_dimension = dimension,
                            autoencoder_type = "RFAE", # Use the radial flattening autoencoder
                        )
                    flattened_points = self.NeuralFlattener.fit_transform(X[cluster_idxs])
                    G_euclidean = self.graph_former(flattened_points)
                    fs = self.unsigned_curvature(G_euclidean, t, idx=0)
                    return fs
                case "Radial Flattener":
                    X_flattened = radially_flatten_with_ae(intrinsic_dim=dimension, X = X[cluster_idxs])
                    G_euclidean = self.graph_former(X_flattened)
                    fs = self.unsigned_curvature(G_euclidean, t, idx=0)
                    return fs

                    

        # Start by estimating the manifold's unsigned curvature, i.e. spreads of diffusion
        manifold_spreads, manifold_spreads_nought, P, Pt, t = self.unsigned_curvature(G,t,idx, _also_return_first_scale=True, D = D)
        if self.verbose: print(f"Manifold spreads are {manifold_spreads}")
        # print(manifold_spreads_nought.shape)
        n = G.W.shape[0]
        if dim is None: # The dimension wasn't supplied; we'll estimate it pointwise
            print("estimating local dimension of each point... may take a while")
            ldims = self.dimest.fit_pw(
                                G.data, #TODO: Currently this requires underlying points!
                                n_neighbors = 100,
                                n_jobs = 1)
            dims_per_point = np.round(ldims.dimension_pw_).astype(int)
        else: # the dimension *was* supplied, but it may be either a single global dimension or a local dimension for each point
            if isinstance(dim, int):
                dims_per_point = jnp.ones(G.W.shape[0], dtype=int)*dim
            else:
                dims_per_point = dim

        if self.flattening_method == "Neural":
            # we need to compute coordinates to flatten. We'll use diffusion maps for this.
            self.diff_coords = diffusion_coordinates(G, t=t)[:,:dim]
        
        flat_spreads = jnp.zeros(n)
        num_points_in_comparison = n // self.comparison_space_size_factor # TODO: Can surely find a better heuristic here
        num_clusters = n // self.points_per_cluster if self.points_per_cluster is not None else 1
        if num_clusters == n: 
            # Construct a separate comparison space for each point
            cluster_labels = jnp.arange(n)
        elif num_clusters == 1:
            # Use just one comparison space for the whole dataset
            cluster_labels = jnp.zeros(n)
        elif idx is not None: 
            cluster_labels = jnp.ones(n) # if a single index is supplied, there's only one cluster.
            cluster_labels = cluster_labels.at[idx].set(0)
            num_clusters = 1
        else: 
            # Cluster dataset into specified num_clusters, construct separate comparison spaces for each.
            cluster_labels = enhanced_spectral_clustering(G, manifold_spreads, dim=dim, num_clusters=num_clusters, )

        for i in range(num_clusters):
            cluster_idxs = jnp.where(cluster_labels==i)[0]
            average_dim_in_cluster = int(jnp.round(jnp.mean(dims_per_point[cluster_idxs])))
            if self.verbose: print(f"{average_dim_in_cluster=}")
            average_spread_in_cluster = jnp.mean(manifold_spreads_nought[cluster_idxs])
            fs = get_flat_spreads(
                dimension = average_dim_in_cluster,
                jump_of_diffusion = average_spread_in_cluster,
                num_points_in_comparison = num_points_in_comparison,
                cluster_idxs = cluster_idxs,
                verbose=True
                )
            # fs = self.unsigned_curvature(G_euclidean,t,idx=0)
            flat_spreads = flat_spreads.at[cluster_idxs].set(
                    fs
                )
        match self.comparison_method:
            case "Ollivier":
                ks = 1 - manifold_spreads/flat_spreads
            case "Subtraction":
                ks = flat_spreads - manifold_spreads
            case _:
                raise ValueError(f'Comparison method must be in {_COMPARISON_METHOD}')    
        if idx is not None: ks = ks[idx]
        return ks #, flat_spreads, manifold_spreads, P, Pt

    def fit(
            self,
            G:pygsp.graphs.Graph, # Input Graph
            t:int, # Scale; if none, finds the knee-point of the spectral entropy curve of the diffusion operator
            idx=None, # the index at which to compute curvature. If None, computes for all points.
            dim = None, # the INTRINSIC dimension of your manifold, as an int for global dimension or list of pointwise dimensions; if none, tries to estimate pointwise.
            knn = 15, # Number of neighbors used in construction of graph;
            D = None, # Supply manifold distances yourself to override their computation. Only used with the Wasserstein laziness method.
            X = None, # if using a flattening method that requires a point cloud, supply it here.
    ):
        self.G = G
        self.X = X
        self.D = D
        self.ks = self.curvature(G = G, t=t, idx=idx, dim=dim, knn=knn, D=D, X=X)

    def fit_transform(
            self,
            G:pygsp.graphs.Graph, # Input Graph
            t:int, # Scale; if none, finds the knee-point of the spectral entropy curve of the diffusion operator
            idx=None, # the index at which to compute curvature. If None, computes for all points.
            dim = None, # the INTRINSIC dimension of your manifold, as an int for global dimension or list of pointwise dimensions; if none, tries to estimate pointwise.
            knn = 15, # Number of neighbors used in construction of graph;
            D = None, # Supply manifold distances yourself to override their computation. Only used with the Wasserstein laziness method.
            X = None, # if using a flattening method that requires a point cloud, supply it here.
    ):
        self.fit(G = G, t=t, idx=idx, dim=dim, knn=knn, D=D, X=X)
        return self.ks
    
    
def fill_diagonal(a, val):
  assert a.ndim >= 2
  i, j = jnp.diag_indices(min(a.shape[-2:]))
  return a.at[..., i, j].set(val)

