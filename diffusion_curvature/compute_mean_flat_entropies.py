__all__ = ['average_flat_entropies', 'create_mean_entropy_database', 'load_average_entropies']

# %% ../nbs/library/core-jax/Mean-Flat-Entropies.ipynb 2
import jax.numpy as jnp
from .core import DiffusionCurvature, get_adaptive_graph
from .utils import *
from tqdm.auto import tqdm
import graphtools

def average_flat_entropies(
        dim,
        t,
        num_trials,
        num_points_in_comparison = 10000,
        graph_former = get_adaptive_graph
):
    DC = DiffusionCurvature(
        laziness_method="Entropic",
        flattening_method="Fixed",
        comparison_method="Subtraction",
        points_per_cluster=None, # construct separate comparison spaces around each point
        comparison_space_size_factor=1
    )
    flat_spreads = jnp.zeros(num_trials)
    for i in range(num_trials):
        Rn = jnp.concatenate([jnp.zeros((1,dim)), 2*random_jnparray(num_points_in_comparison-1, dim)-1])
        G = graph_former(Rn) #graphtools.Graph(Rn, anisotropy=1, knn=k, decay=None,).to_pygsp()
        fs = DC.unsigned_curvature(G, t, idx=0)
        flat_spreads = flat_spreads.at[i].set(fs)
    return jnp.mean(flat_spreads)

# %% ../nbs/library/core-jax/Mean-Flat-Entropies.ipynb 3
import h5py
from fastcore.all import *
@call_parse
def create_mean_entropy_database(
    outfile = "../data/entropies_averaged.h5",
    dimensions:Param("(Intrinsic) Dimensions to Take Mean Entropies over", int, nargs='+') = [1,2,3,4,5,6],
    knns:Param("k-nearest neighbor values to compute", int, nargs='+') = [5,10,15],
    ts:Param("time values to compute", int, nargs='+') = [25,30,35],
):
    # load the database
    f = h5py.File(outfile,'a')
    for i, dim in tqdm(enumerate(dimensions)):
        # load the group corresponding to dimension; create if it doesn't exist
        if str(dim) in f.keys(): dim_group = f[str(dim)]
        else:               dim_group = f.create_group(str(dim))
        for j, knn in tqdm(enumerate(knns),leave=False):
            if str(knn) in dim_group.keys(): knn_group = dim_group[str(knn)]
            else:                            knn_group = dim_group.create_group(str(knn))
            for k, t in tqdm(enumerate(ts),leave=False):
                if str(t) in knn_group.keys(): continue
                else:
                    afe = average_flat_entropies(dim, knn, t, 100)
                    knn_group.create_dataset(str(t), data=afe)
    return f


# %% ../nbs/library/core-jax/Mean-Flat-Entropies.ipynb 4
import h5py
def load_average_entropies(filename):
    d = {}
    with h5py.File(filename,'r') as f:
        for dim in f.keys():
            d[dim] = {}
            for knn in f[dim].keys():
                d[dim][knn] = {}
                for t in f[dim][knn].keys():
                    d[dim][knn][t] = f[dim][knn][t][()]
    return d
