# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/0e-Errata.ipynb.

# %% auto 0
__all__ = ['jax_set_best_gpu', 'kronecker_delta', 'plot_3d', 'perform_trials', 'plot_array', 'random_jnparray',
           'visualize_embedding', 'printnum']

# %% ../nbs/0e-Errata.ipynb 3
import jax
import jax.numpy as jnp

def jax_set_best_gpu():
    devices = jax.devices("gpu")
    if not devices:
        return None

    max_memory_gpu_idx = None
    max_memory = 0

    for idx, device in enumerate(devices):
        device_mem = device.memory_stats()
        free_memory = device_mem['bytes_limit'] - device_mem['bytes_in_use']
        if free_memory > max_memory:
            max_memory = free_memory
            max_memory_gpu_idx = idx

    if max_memory_gpu_idx is not None:
        jax.config.update("jax_default_device", devices[max_memory_gpu_idx])
    
    return max_memory_gpu_idx

# %% ../nbs/0e-Errata.ipynb 5
import numpy as np
def kronecker_delta(
        length, # length of array. If you pass an array or list, length is set to size of the first dimension.
        idx=None # idx to make nonzero. If none, fills a random idx.
        ):
    """ returns np array of len with all zeroes except idx. """
    if not isinstance(length,int): length = len(length)
    if idx is None: idx = np.random.randint(0,length)
    x = np.zeros(length)
    x[idx]=1
    return x

# %% ../nbs/0e-Errata.ipynb 6
# For plotting 2D and 3D graphs
import plotly
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

def plot_3d(X,distribution=None, title="",lim=None,use_plotly=False, zlim = None, colorbar = False, cmap="plasma"):
    if distribution is None:
        distribution = np.zeros(len(X))
    if lim is None:
        lim = np.max(np.linalg.norm(X,axis=1))
    if zlim is None:
        zlim = lim
    if use_plotly:
        d = {'x':X[:,0],'y':X[:,1],'z':X[:,2],'colors':distribution}
        df = pd.DataFrame(data=d)
        fig = px.scatter_3d(df, x='x',y='y',z='z',color='colors', title=title, range_x=[-lim,lim], range_y=[-lim,lim],range_z=[-zlim,zlim])
        fig.show()
    else:
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(111,projection='3d')
        ax.axes.set_xlim3d(left=-lim, right=lim)
        ax.axes.set_ylim3d(bottom=-lim, top=lim)
        ax.axes.set_zlim3d(bottom=-zlim, top=zlim)
        im = ax.scatter(X[:,0],X[:,1],X[:,2],c=distribution,cmap=cmap)
        ax.set_title(title)
        if colorbar: fig.colorbar(im, ax=ax)
        plt.show()

# %% ../nbs/0e-Errata.ipynb 7
from fastcore.all import *
from functools import partial
from functools import wraps
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import time
from multiprocessing import Process,Queue,Manager,set_start_method,get_all_start_methods,get_context
from threading import Thread
from tqdm.notebook import trange, tqdm


def perform_trials(
                f,
                n_trials=10, 
                n_workers=defaults.cpus - 2, 
                pause=0,
                method=None,
                threadpool=False,
                **kwargs):
    """An adaptation of fastcore's parallel for running the same function multiple times."""
    futures = []
    results = []
    with ProcessPoolExecutor(n_workers) as ex:
        for i in range(n_trials): 
            futures.append(ex.submit(f, **kwargs))
        for future in tqdm(as_completed(futures)):
            results.append(future.result())
    return results

# %% ../nbs/0e-Errata.ipynb 9
import seaborn as sns
import matplotlib.pyplot as plt

def plot_array(ratios, xs=None, title=""):
    sns.set(style="darkgrid")  # Set the seaborn style

    # Create a figure and axes
    fig, ax = plt.subplots()
    # Compute the mean and standard deviation for the estimated array
    mean_estimated = np.mean(ratios, axis=0)
    std_estimated = np.std(ratios, axis=0)

    # Plot estimated as a line with error bars
    if xs is None:
        ax.errorbar(np.arange(ratios.shape[1]), mean_estimated, yerr=std_estimated, label="Estimated")
    else:
        ax.errorbar(xs, mean_estimated, yerr=std_estimated, label="Estimated")

    # Set the labels for x-axis and y-axis
    ax.set_xlabel("Radius")
    ax.set_ylabel("Volume of n-sphere")

    # Set the title of the plot
    ax.set_title(title)

    # Display the legend
    ax.legend()

    # Show the plot
    plt.show()

# %% ../nbs/0e-Errata.ipynb 10
# random array in jax
import jax.random
import random
def random_jnparray(*shape):
    key = random.randint(0,10000)
    rng = jax.random.PRNGKey(key)
    rng, subkey1, subkey2 = jax.random.split(rng, num=3)
    rand_array = jax.random.uniform(subkey1,shape=shape)
    return rand_array

# %% ../nbs/0e-Errata.ipynb 11
import numpy as np
import matplotlib.pyplot as plt
# visualize the latent embedding space of a pytorch model, colored by a given vector
def visualize_embedding(
    model,
    dataloader,
    colors=None,
    title = "Embedded Points"
):
    if colors is None:
        colors = np.zeros(len(dataloader.dataset))
    model.eval()

    embeddings = model.encoder(dataloader.dataset.pointcloud).cpu().detach().numpy()
    
    plt.figure(figsize=(10, 10))
    plt.scatter(embeddings[:, 0], embeddings[:, 1], c=colors, cmap="viridis")
    plt.title(title)
    plt.colorbar()
    plt.show()

# %% ../nbs/0e-Errata.ipynb 12
def printnum(number):
    suffixes = ['th', 'st', 'nd', 'rd', 'th', 'th', 'th', 'th', 'th', 'th']
    if 10 <= number % 100 <= 20:
        suffix = 'th'
    else:
        suffix = suffixes[number % 10]
    return f"{number}{suffix}"
