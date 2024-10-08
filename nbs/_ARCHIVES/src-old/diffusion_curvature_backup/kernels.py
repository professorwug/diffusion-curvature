# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/library/Kernels.ipynb.

# %% auto 0
__all__ = ['median_heuristic', 'gaussian_kernel', 'pygsp_graph_from_points', 'knn_graph', 'diffusion_matrix', 'plot_3d',
           'compute_anisotropic_affinities_from_graph', 'compute_anisotropic_diffusion_matrix_from_graph']

# %% ../../nbs/library/Kernels.ipynb 5
import numpy as np
def median_heuristic(
        D:np.ndarray, # the distance matrix
):
    # estimate kernel bandwidth from distance matrix using the median heuristic
    # Get upper triangle from distance matrix (ignoring duplicates)
    h = D[np.triu_indices_from(D)]
    h = h**2
    h = np.median(h)
    nu = np.sqrt(h / 2)
    return nu

# %% ../../nbs/library/Kernels.ipynb 6
import numpy as np
from sklearn.metrics import pairwise_distances
def gaussian_kernel(
        X:np.ndarray, # pointcloud data as rows, shape n x d
        kernel_type = "fixed", # either fixed, or adaptive
        sigma:float = 0, # if fixed, uses kernel bandwidth sigma. If not set, uses a heuristic to estimate a good sigma value
        k:float = 10, # if adaptive, creates a different kernel bandwidth for each point, based on the distance from that point to the kth nearest neighbor
        anisotropic_density_normalization:float = 0.5, # if nonzero, performs anisotropic density normalization
        threshold_for_small_values:float = 1e-5, # Sets all affinities below this value to zero. Set to zero to disable.
):
    """Constructs an affinity matrix from pointcloud data, using a gaussian kernel"""
    supported_kernel_types = {'fixed', 'adaptive'}
    assert kernel_type in supported_kernel_types
    D = pairwise_distances(X)
    if kernel_type == "fixed":
            if not sigma:
                # estimate sigma using a heuristic
                sigma = median_heuristic(D)
            W = (1/(sigma*np.sqrt(2*np.pi)))*np.exp((-D**2)/(2*sigma**2))
    elif kernel_type == "adaptive":
            distance_to_k_neighbor = np.partition(D,k)[:,k]
            # Populate matrices with this distance for easy division.
            div1 = np.ones(len(D))[:,None] @ distance_to_k_neighbor[None,:]
            div2 = distance_to_k_neighbor[:,None] @ np.ones(len(D))[None,:]
            # print("Distance to kth neighbors",distance_to_k_neighbor)
            # compute the gaussian kernel with an adaptive bandwidth
            W = (1/(2*np.sqrt(2*np.pi)))*(np.exp(-D**2/(2*div1**2))/div1 + np.exp(-D**2/(2*div2**2))/div2)
    if anisotropic_density_normalization:
        D = np.diag(1/(np.sum(W,axis=1)**anisotropic_density_normalization))
        W = D @ W @ D
    if threshold_for_small_values:
        W[W < threshold_for_small_values] = 0
    return W

# %% ../../nbs/library/Kernels.ipynb 7
import pygsp
def pygsp_graph_from_points(X, knn=15):
    W = gaussian_kernel(X, kernel_type="adaptive", k=knn, anisotropic_density_normalization=1)
    G = pygsp.graphs.Graph(W)
    return G

# %% ../../nbs/library/Kernels.ipynb 9
from sklearn.neighbors import kneighbors_graph
def knn_graph(
        X:np.ndarray,
        k = 10,
        pygsp = True,
):
    A = kneighbors_graph(X, k, mode='connectivity', include_self=True)
    if pygsp:
        G = pygsp.graphs.Graph(W)
    else:
        return A

# %% ../../nbs/library/Kernels.ipynb 11
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize
from graphtools.matrix import set_diagonal

def diffusion_matrix(
        X:np.ndarray = None, # pointcloud data
        A:np.ndarray = None, # adjacency matrix, if precomputed
        kernel_type:str = "fixed", # either fixed or adaptive
        sigma = 0, # if fixed, uses kernel bandwidth sigma. If not set, uses a heuristic to estimate a good sigma value
        k = 10, # if adaptive, creates a different kernel bandwidth for each point, based on the distance from that point to the kth nearest neighbor
        anisotropic_density_normalization = 0.5, # if nonzero, performs anisotropic density normalization
        threshold_for_small_values = 1e-5,
):
    """ Creates a diffusion matrix from pointcloud data, by row-normalizing the affinity matrix obtained from the gaussian_kernel function """
    if X is not None:
        W = gaussian_kernel(X,kernel_type,sigma=sigma,k = k,anisotropic_density_normalization = anisotropic_density_normalization, threshold_for_small_values=threshold_for_small_values)
        W = W + np.eye(len(X))*1e-5
    if X is None and A is not None:
        W = A
    
    K = set_diagonal(W, 1)
    P = normalize(W, norm="l1", axis=1)
    return P

# %% ../../nbs/library/Kernels.ipynb 24
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

# %% ../../nbs/library/Kernels.ipynb 43
import numpy as np
def compute_anisotropic_affinities_from_graph(
    A:np.ndarray, # the adjacency/affinity matrix of the graph
    alpha:float, # the anisotropic density normalization parameter
) -> np.ndarray:
    # normalize by density
    D = np.diag(1/np.sum(A,axis=1)**alpha)
    A_anis = D @ A @ D
    return A_anis

def compute_anisotropic_diffusion_matrix_from_graph(
    A:np.ndarray, # the adjacency/affinity matrix of the graph
    alpha:float, # the anisotropic density normalization parameter
    ) -> np.ndarray:
    A_anis = compute_anisotropic_affinities_from_graph(A,alpha)
    # row normalize to create diffusion matrix
    D = np.diag(1/(np.sum(A_anis,axis=1)+1e-8))
    P = D @ A_anis
    return P
