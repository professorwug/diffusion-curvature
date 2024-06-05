
# %% auto 0
__all__ = ['get_average_local_distances', 'scale_by_local_distances']

# %% ../nbs/experiments/3h1-local-distance-scaling.ipynb 5
import jax.numpy as jnp
from sklearn.metrics import pairwise_distances
from fastcore.all import *

# def partition_by_row(arr, k):
#     return jnp.apply_along_axis(lambda x: jnp.partition(x, k), arr=arr, axis=1)

def get_average_local_distances(X, k = 1):
    D = pairwise_distances(X)
    nearest_distances = jnp.partition(D, k, axis = 1)[:,k]
    average_nearest_distances = jnp.median(nearest_distances)
    return  average_nearest_distances, nearest_distances

def scale_by_local_distances(X_manifold, X_comparison, k = 1):
    m_average_nearest_distances, m_nearest_distances = get_average_local_distances(X_manifold, k = k)
    c_average_nearest_distances, c_nearest_distances = get_average_local_distances(X_comparison, k = k)
    
    distance_ratio = m_average_nearest_distances / c_average_nearest_distances
    
    return X_comparison * distance_ratio
