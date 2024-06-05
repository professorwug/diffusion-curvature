__all__ = ['create_battery', 'main']

# %% ../nbs/library/datasets/Construct-Battery.ipynb 2
from tqdm.auto import trange, tqdm
from .random_surfaces import samples_from_random_surface
from functools import partial
def create_battery(
        intrinsic_dims = [3,4,5,6],
        codimensions = [1,2],
        num_manifolds_per_dim = 10,
        noise_levels = [0.0, 0.05, 0.1,0.2,0.3,0.5],
):
    CC = {}
    CC['dims'] = intrinsic_dims
    for d in tqdm(intrinsic_dims, desc="Intrinsic Dimension"):
        n_samples = 1000*2**d if d > 3 else 5000 # TODO: Rough heuristic
        CC[d] = {}
        CC[d]['codims'] = codimensions
        for c in codimensions:
            CC[d][c] = {}
            CC[d][c]['noise_levels'] = noise_levels
            for noise_level in tqdm(noise_levels, desc="Noise Level", leave=False):
                CC[d][c][noise_level] = {}
                CC[d][c][noise_level]["Xs"] = []
                CC[d][c][noise_level]["k"] = []
                N = d+c
                for i in trange(num_manifolds_per_dim, leave=False):
                    X, k = samples_from_random_surface(n_samples, d, N, degree=2, noise_level=noise_level)
                    # Turns out doing it in parallel messes with the random number generator
                    # specified_sampler = partial(samples_from_random_surface, n_samples, d, N, degree=2, noise_level=noise_level)
                    # results = perform_trials()
                    CC[d][c][noise_level]["Xs"].append(X)
                    CC[d][c][noise_level]["k"].append(k)
    return CC

# %% ../nbs/library/datasets/Construct-Battery.ipynb 3
import os
from fastcore.all import *
import deepdish

@call_parse
def main(
        filename:str="/home/piriac/data/diffusion_curvature/Curvature_Colosseum.h5", # path to the sampled toy manifolds
        intrinsic_dims:list=[2,3,4,5], # intrinsic dimensions of the toy manifolds
        codimensions:list=[1,2,3,4], # codimensions of the toy manifolds
        num_manifolds_per_dim:int=50, # number of toy manifolds per intrinsic dimension
        noise_levels:list = [0.0, 0.05, 0.1,0.2,0.3,0.5],

):
    """Constructs a battery of toy manifolds and saves it to disk in HDF5 format"""
    print(f"Sampling from {num_manifolds_per_dim} manifolds in dims {intrinsic_dims} with codims {codimensions} and noise level {noise_levels}. Saving to {filename}")
    CC = create_battery(
        intrinsic_dims=intrinsic_dims,
        codimensions = codimensions,
        num_manifolds_per_dim=num_manifolds_per_dim,
        noise_levels=noise_levels,
    )
    deepdish.io.save(filename, CC)
    
    
