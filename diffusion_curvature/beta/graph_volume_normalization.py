
# %% auto 0
__all__ = []

# %% ../../nbs/experiments/2c5-graph-volume-normalization.ipynb 8
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from fastcore.all import *
from ..core import *
import jax.numpy as jnp

@patch 
def graph_volumes(self:DiffusionCurvature, epsilon_threshold = 1e-5):
    """
    returns the graph volumes across all idxs. Presumes that Pt is precomputed.
    """
    idxs_above_threshold = (self.Pt > epsilon_threshold).astype(int)
    graph_volumes = jnp.sum(idxs_above_threshold, axis=-1)
    return graph_volumes

@patch
def division_normalized_laziness(self:DiffusionCurvature, G, t, epsilon_threshold, return_all = "norm"):
    laziness = self.unsigned_curvature(G, t)
    volumes = self.graph_volumes(epsilon_threshold = epsilon_threshold)
    norm = laziness / volumes
    if return_all == "all": return norm, laziness, volumes
    elif return_all == "volume": return volumes
    else: return norm
