# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/0c1a-Diffusion-Laziness.ipynb.

# %% auto 0
__all__ = ['wasserstein_spread_of_diffusion', 'entropy_of_diffusion', 'kl_div', 'js_dist', 'diffusion_distances_along_trajectory',
           'trapezoidal_rule', 'DiffusionLaziness', 'curvature_curves', 'compare_curvature_across_datasets',
           'compare_curvature_across_datasets_by_maximum_mean_discrepancy']

# %% ../nbs/0c1a-Diffusion-Laziness.ipynb 5
import jax
import jax.numpy as jnp
from jax import jit
import scipy

@jit
def wasserstein_spread_of_diffusion(
                D:jax.Array, # manifold geodesic distances
                Pt:jax.Array, # powered diffusion matrix/t-step ehat diffusions
                ):
        """
        Returns how "spread out" each diffusion is, with wasserstein distance
        Presumes that the manifold distances have been separately calculated
        """
        return jnp.sum(D * Pt, axis=-1)

# %% ../nbs/0c1a-Diffusion-Laziness.ipynb 13
import jax.scipy
import jax.numpy as jnp

def entropy_of_diffusion(
    Pt:jax.Array, # powered diffusion matrix
    epsilon=1e-5, # threshold for small values, for speed
): 
        """
        Returns the pointwise entropy of diffusion from the powered diffusion matrix in the input
        Assumes that Pt sums to 1
        """
        # Use only the elements of Pt that are greater than epsilon
        Pt = Pt * (Pt>epsilon)
        # Normalize Pt so that it sums to 1
        Pt = Pt / (jnp.sum(Pt, axis=-1) + 1e-12)
        # Pt = (Pt + 1e-10) /(1 + 1e-10*Pt.shape[0]) # ensure, for differentiability, that there are no zeros in Pt, but that it still sums to 1.
        entropy_elementwise = jax.scipy.special.entr(Pt)
        entropy_of_rows = jnp.sum(entropy_elementwise, axis=-1)
        # normalize so max value is 1
        # entropy_of_rows = entropy_of_rows / (-jnp.log(1/jnp.sum(Pt>epsilon, axis=-1)))
        return entropy_of_rows

# %% ../nbs/0c1a-Diffusion-Laziness.ipynb 22
@jax.jit
def kl_div(A, B, eps = 1e-12):
    # Calculate Kullback-Leibler divergence
    # get rid of zero values
    A = jnp.where(A == 0, eps, A)
    B = jnp.where(B == 0, eps, B)
    v = A*(jnp.log(A) - jnp.log(B)) 
    return jnp.sum(v)

# %% ../nbs/0c1a-Diffusion-Laziness.ipynb 24
@jax.jit
def js_dist(
    P:jax.Array, 
    Q:jax.Array,
):
    """Compute the Jensen-Shannon distance between two probability distributions.

    Input
    -----
    P, Q : array-like
        Probability distributions of equal length that sum to 1
    """

    M = 0.5 * (P + Q)

    # Get the JS DIVERGENCE
    result = 0.5 * (kl_div(P, M) + kl_div(Q, M))
    # Take sqrt to get the JS DISTANCE
    return jnp.sqrt(jnp.abs(result))

# %% ../nbs/0c1a-Diffusion-Laziness.ipynb 32
from scipy.spatial.distance import jensenshannon
def diffusion_distances_along_trajectory(diffusions):
    # given a sequence of diffusions, returns the distances between each 
    js_dist_vectorized = jax.vmap(js_dist, (0, 0), 0)
    distances = [jnp.zeros(diffusions[0].shape[0])]
    for idx in range(len(diffusions)-1):
        step_distance = js_dist_vectorized(diffusions[idx+1], diffusions[idx])
        distances.append(
            distances[-1] + step_distance
        )
    return jnp.stack(distances)

# %% ../nbs/0c1a-Diffusion-Laziness.ipynb 36
import jax
import jax.numpy as jnp

@jax.jit
def trapezoidal_rule(x, y):
    # Ensure x and y are JAX arrays
    x = jnp.asarray(x)
    y = jnp.asarray(y)
    
    # Calculate the differences between consecutive x values along the second axis (axis=1)
    dx = x[:, 1:] - x[:, :-1]
    
    # Calculate the trapezoidal areas along the second axis, handling NaNs
    trapezoidal_areas = dx * (y[:, :-1] + y[:, 1:]) / 2
    
    # Mask out the NaNs in the trapezoidal areas
    valid_mask = ~jnp.isnan(trapezoidal_areas)
    trapezoidal_areas = jnp.where(valid_mask, trapezoidal_areas, 0.0)
    
    # Sum up the areas along the second axis to get the integral for each row
    integral = jnp.sum(trapezoidal_areas, axis=1)
    
    return integral

# %% ../nbs/0c1a-Diffusion-Laziness.ipynb 38
from typing import Literal
from .kernels import diffusion_matrix_from_affinities
from .heat_diffusion import heat_diffusion_from_dirac, powers_of_diffusion
from copy import deepcopy

class DiffusionLaziness():
    DIFFUSION_TYPES = Literal['diffusion matrix','heat kernel']
    LAZINESS_METHODS = Literal['Entropic', 'Wasserstein']
    def __init__(
        self,
        diffusion_type:DIFFUSION_TYPES = "diffusion matrix",
        laziness_method:LAZINESS_METHODS = "Entropic",
        smoothing = 2, 
    ):
        store_attr()

    def fit_transform(
        self,
        G, # graph
        ts, # time or list of times.
        idx = None, # supply an integer or list of indices, and we'll only calculate their laziness
        D = None,
        t_dist:int = 25, # diffusion time for distance calculation
    ):
        # get jax affinity matrix, and compute diffusion matrix from graph
        W = G.W
        if scipy.sparse.issparse(W):
            W = W.todense()
        ts = deepcopy(ts)
        if isinstance(ts, int): ts = [ts]
        if D is None: ts += [t_dist]
        if self.smoothing: ts = [self.smoothing] + ts
        W = jnp.array(W)
        # get powers of diffusion
        match self.diffusion_type:
            case 'diffusion matrix':
                P = diffusion_matrix_from_affinities(W)
                Pts = powers_of_diffusion(P, ts)
            case 'heat kernel':
                raise NotImplementedError # TODO: Implement and test
                Pts = heat_diffusion_from_dirac(G, ts)
        match self.laziness_method:
            case "Wasserstein":
                if D is None: D = phate_distances(Pts[-1])
                laziness_with_distance = partial(wasserstein_spread_of_diffusion, D = D)
                laziness_fn = jax.vmap(wasserstein_spread_of_diffusion)
            case "Entropic":
                laziness_fn = jax.vmap(entropy_of_diffusion, (0), 0)
        self.ts = ts
        diffusions = Pts

        if D is None: 
            diffusions = diffusions[:-1] # the last Pt is for heat 
            self.ts = self.ts[:-1]
        if self.smoothing is not None:
            self.smoothing_P = diffusions[0]
            diffusions = diffusions[1:]
            self.ts = self.ts[1:]


        if idx is not None: diffusions = [d[idx][None,:] for d in diffusions]

        self.ls = laziness_fn(jnp.stack(diffusions)).T 
        self.ds = diffusion_distances_along_trajectory(diffusions).T
        if len(self.ts) > 1: laziness_under_curve = trapezoidal_rule(self.ds, self.ls)
        else:                laziness_under_curve = self.ls

        if self.smoothing and idx is None: # TODO there are probably more intelligent ways to do this smoothing
            # Local averaging to counter the effects local density
            average_laziness = self.smoothing_P @ laziness_under_curve
            laziness_under_curve = average_laziness.squeeze()
        return laziness_under_curve
    
    def integrate_laziness_with_bounds(self, min_d, max_d, idxs=None):
        if idxs is None:
            idxs = jnp.arange(self.ds.shape[0])
        else:
            idxs = jnp.asarray(idxs)
        
        ds_filtered = self.ds[idxs]
        ls_filtered = self.ls[idxs]
        
        mask = (ds_filtered >= min_d) & (ds_filtered <= max_d)
        valid_ds = jnp.where(mask, ds_filtered, jnp.nan)
        valid_ls = jnp.where(mask, ls_filtered, jnp.nan)
        
        results = trapezoidal_rule(valid_ds, valid_ls)

        return results
        

# %% ../nbs/0c1a-Diffusion-Laziness.ipynb 68
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from fastcore.all import *
from tqdm.auto import tqdm
import inspect

def curvature_curves(*diffusion_curvatures, idx=0, title="Curvature Curves", also_plot_against_time = True, **kwargs):
    if also_plot_against_time: fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    else:                      fig, axs = plt.subplots(1, 1, figsize=(6, 6))
    for dc in diffusion_curvatures:
        dc_name = None
        for frame_record in inspect.stack():
            frame = frame_record.frame
            for name, obj in frame.f_globals.items():
                if obj is dc:
                    dc_name = name
            for name, obj in frame.f_locals.items():
                if obj is dc:
                    dc_name = name
        if dc_name is None:
            dc_name = "Unknown"
        t_values, distances, curvatures = dc.ts, dc.ds[idx], dc.ls[idx]
        # else:
        #     t_values, distances, curvatures = dc.manifold_lazy_est.ts, dc.manifold_lazy_est.ds[idx], dc.manifold_lazy_est.ls[idx]
        axs[1].plot(distances, curvatures, label=dc_name)
        if also_plot_against_time: axs[0].plot(t_values, curvatures, label=dc_name)
        
    axs[1].set_title("Diffusion Energy Vs. Diffusion Trajectory Distance")
    axs[1].set_xlabel('Distance')
    axs[1].set_ylabel('Diffusion Energy')
    if also_plot_against_time:
        axs[0].set_title("Diffusion Energy vs. Time")
        axs[0].set_xlabel('Time ($t$)')    
        axs[0].set_ylabel('Diffusion Energy')
        axs[0].legend()
    fig.suptitle(title)
    axs[1].legend()
    plt.show()

# %% ../nbs/0c1a-Diffusion-Laziness.ipynb 73
def compare_curvature_across_datasets(
    *diffusion_lazinesses,
    idxs:List  = None # list of idxs to compare. Can also be a list of lists of idxs, one per DiffusionLaziness
):
    if idxs is None:
        idxs = jnp.arange(diffusion_lazinesses[0].ds.shape[0])
    def _get_means_by_idxs(DL, id):
        idxs = jnp.array(id)
        if len(idxs) == 1:
            mean_ds = DL.ds[id[0]] 
        else:
            mean_ds = jnp.mean(DL.ds[id], axis=0)
        return mean_ds

    if isinstance(idxs[0], List) or isinstance(idxs[0], jax.Array):
        assert len(idxs) == len(diffusion_lazinesses)
        mean_ds = [_get_means_by_idxs(diffusion_lazinesses[i],idxs[i]) for i in range(len(diffusion_lazinesses))]
    else:
        mean_ds = [_get_means_by_idxs(DL,idxs) for DL in diffusion_lazinesses]
    
    max_ds = jnp.array([jnp.max(md) for md in mean_ds])
    minmax_ds = jnp.min(max_ds)
    if isinstance(idxs[0], List) or isinstance(idxs[0], jax.Array):
        bounded_integrals = [diffusion_lazinesses[i].integrate_laziness_with_bounds(0, minmax_ds, idxs=idxs[i]) for i in range(len(diffusion_lazinesses))]
    else:
        bounded_integrals = [DL.integrate_laziness_with_bounds(0, minmax_ds, idxs = idxs) for DL in diffusion_lazinesses]

    return bounded_integrals

# %% ../nbs/0c1a-Diffusion-Laziness.ipynb 77
def compare_curvature_across_datasets_by_maximum_mean_discrepancy(
    target_laziness:DiffusionLaziness, # the DiffusionLaziness operator of the manifold
    comparison_laziness:DiffusionLaziness,
    idxs:List, # the DiffusionLaziness operator of the comparison space 
    method: str = "quadratic" # method can be "quadratic" or "piecewise"
):
    assert method in ["quadratic", "piecewise"], "method must be 'quadratic' or 'piecewise'"

    from jax.scipy.linalg import lstsq

    def quadratic_regression(x, y):
        # Fit a quadratic polynomial y = ax^2 + bx + c
        A = jnp.vstack([x**2, x, jnp.ones_like(x)]).T
        coeffs, _, _, _ = lstsq(A, y)
        return coeffs

    def evaluate_quadratic(coeffs, x):
        return coeffs[0] * x**2 + coeffs[1] * x + coeffs[2]

    # Extract the ds and ls from each, treating them as x and y coordinates to be modeled
    target_x = target_laziness.ds[idxs] # has shape num_idxs x num_ts
    target_y = target_laziness.ls[idxs] 
    comp_x = comparison_laziness.ds
    comp_y = comparison_laziness.ls

    if method == "quadratic":
        # Perform quadratic regression on these pairs, creating num_idxs functions target_fns, comp_fns
        target_coeffs = jax.vmap(quadratic_regression)(target_x, target_y)
        comp_coeffs = quadratic_regression(comp_x, comp_y)

        # Create a range of x values to evaluate the polynomials
        x_values = jnp.linspace(jnp.min(target_x), jnp.max(target_x), 500)

        # Evaluate the polynomials
        target_fns = jax.vmap(lambda coeffs: evaluate_quadratic(coeffs, x_values))(target_coeffs)
        comp_fn = evaluate_quadratic(comp_coeffs, x_values)
    else:
        # Use piecewise linear approximation
        x_values = jnp.unique(jnp.concatenate([target_x.flatten(), comp_x]))
        target_fns = jax.vmap(lambda x, y: jnp.interp(x_values, x, y))(target_x, target_y)
        comp_fn = jnp.interp(x_values, comp_x, comp_y)

    if method == "quadratic":
        # Find the x value where the mean absolute discrepancy is greatest
        discrepancies = jnp.abs(target_fns - comp_fn)
        mean_discrepancies = jnp.mean(discrepancies, axis=0)
        max_discrepancy_idx = jnp.argmax(mean_discrepancies)
        max_discrepancy_x = x_values[max_discrepancy_idx]

        # Return the associated y values
        return max_discrepancy_x, target_fns[:, max_discrepancy_idx], comp_fn[max_discrepancy_idx]
    else:
        # Find the x value where the mean absolute discrepancy is greatest
        discrepancies = jnp.abs(target_fns - comp_fn)
        mean_discrepancies = jnp.mean(discrepancies, axis=0)
        max_discrepancy_idx = jnp.argmax(mean_discrepancies)
        max_discrepancy_x = x_values[max_discrepancy_idx]

        # Return the associated y values
        return max_discrepancy_x, target_fns[:, max_discrepancy_idx], comp_fn[max_discrepancy_idx]
    
    
