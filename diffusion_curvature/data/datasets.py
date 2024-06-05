# %% auto 0
__all__ = ['rejection_sample_for_torus', 'torus', 'rejection_sample_for_hyperboloid', 'hyperboloid',
           'rejection_sample_for_ellipsoid', 'ellipsoid', 'sphere', 'rejection_sample_for_saddle', 'paraboloid',
           'rejection_sample_from_saddle', 'plane']

# %% ../nbs/library/datasets/toy-datasets.ipynb 2
import numpy as np
from .utils import plot_3d
from nbdev.showdoc import *

# %% ../nbs/library/datasets/toy-datasets.ipynb 16
import numpy as np
def rejection_sample_for_torus(n, r, R):
    # Rejection sampling torus method [Sampling from a torus (Revolutions)](https://blog.revolutionanalytics.com/2014/02/sampling-from-a-torus.html)
    xvec = np.random.random(n) * 2 * np.pi
    yvec = np.random.random(n) * (1/np.pi)
    fx = (1 + (r/R)*np.cos(xvec)) / (2*np.pi)
    return xvec[yvec < fx]

def torus(n=2000, c=2, a=1, noise=None, seed=None, use_guide_points = False):
    """
    Sample `n` data points on a torus. Modified from [tadasets.shapes — TaDAsets 0.1.0 documentation](https://tadasets.scikit-tda.org/en/latest/_modules/tadasets/shapes.html#torus)
    Uses rejection sampling.

    In addition to the randomly generated points, a few constant points have been added.
    The 0th point is on the outer rim, in a region of high positive curvature. The 1st point is in the inside, in a region of negative curvature, and the 2nd point is on the top, where the curvature should be closer to zero.

    Parameters
    -----------
    n : int
        Number of data points in shape.
    c : float
        Distance from center to center of tube.
    a : float
        Radius of tube.
    ambient : int, default=None
        Embed the torus into a space with ambient dimension equal to `ambient`. The torus is randomly rotated in this high dimensional space.
    seed : int, default=None
        Seed for random state.
    """

    assert a <= c, "That's not a torus"
    if use_guide_points: n = n-3

    np.random.seed(seed)
    theta = np.empty(0)
    while len(theta) < n:
        theta = np.append(theta, rejection_sample_for_torus(100, a, c))
    theta = theta[:n]
    # theta = rejection_sample_for_torus(n-2, a, c)
    phi = np.random.random((len(theta))) * 2.0 * np.pi

    data = np.zeros((len(theta), 3))
    data[:, 0] = (c + a * np.cos(theta)) * np.cos(phi)
    data[:, 1] = (c + a * np.cos(theta)) * np.sin(phi)
    data[:, 2] = a * np.sin(theta)

    if use_guide_points:
        data = np.vstack([[[0,-c-a,0],[0,c-a,0],[0,c,a]],data])

    X = data
    if noise:
        noise = np.random.normal(size = X.shape, loc = 0, scale = noise)
        X = X + noise

    # compute curvature of sampled torus
    ks = 8*np.cos(theta)/(5 + np.cos(theta))

    return X, ks

# %% ../nbs/library/datasets/toy-datasets.ipynb 25
def rejection_sample_for_hyperboloid(n,a,b,c,u_limit):
    theta = np.random.random(n)*2*np.pi
    u = (np.random.random(n)*2 - 1)*u_limit
    fx = np.sqrt(a**2 * b**2 * u**2 + a**2 * u**2 * np.sin(theta)**2 + a**2 * np.sin(theta)**2  - b**2 * u**2 * np.sin(theta)**2 + b**2 * u**2 - b**2 * np.sin(theta)**2 + b**2) 
    yvec = np.random.random(n) * (1/np.max(fx))
    return theta[yvec < fx], u[yvec < fx]

def hyperboloid(n=2000,a=2,b=2,c=1, u_limit = 2, seed=None, noise = 0):
    """Sample roughly n points on a hyperboloid, using rejection sampling.

    Parameters
    ----------
    n : int, optional
        number of points, by default 2000
    a : int, optional
        hyperboloid param1, by default 2
    b : int, optional
        hyperboloid param2, by default 2
    c : int, optional
        stretchiness in z, by default 1
    u_limit : int, optional
        Constrain the free parameter u to [-l,l], by default 2
    seed : int, optional
        For repeatability, seed the randomness, by default None

    Returns
    -------
    The sampled points, and the curvatures of each point
    """

    np.random.seed(seed)
    theta, u = rejection_sample_for_hyperboloid(n,a,b,c,u_limit)
    data = np.zeros((len(theta), 3))
    data[:, 0] = a*np.cos(theta)*np.sqrt(u**2 + 1)
    data[:, 1] = b*np.sin(theta)*np.sqrt(u**2 + 1)
    data[:, 2] = c*u

    # compute curvature of sampled hyperboloid
    ks = -(2/(5*data[:,2]**2 + 1)**2)

    X = data
    if noise:
        noise = np.random.normal(size = X.shape, loc = 0, scale = noise)
        X = X + noise

    return X, ks

# %% ../nbs/library/datasets/toy-datasets.ipynb 31
def rejection_sample_for_ellipsoid(n,a,b,c):
    theta = np.random.random(n)*2*np.pi
    phi = np.random.random(n)*2*np.pi
    fx = np.sqrt(-a**2 * b**2 * np.sin(phi)**4 + a**2 * b**2 * np.sin(phi)**2 + a**2 * c**2 * np.sin(phi)**4 * np.sin(theta)**2 - b**2 * c**2 * np.sin(phi)**4 * np.sin(theta)**2 + b**2 * c**2 * np.sin(phi)**4)
    yvec = np.random.random(n) * (1/np.max(fx))
    return theta[yvec < fx], phi[yvec < fx]

def _ellipsoid_density_defects(n=2000,a=3,b=2,c=1, seed=None, noise=None):
    """Sample roughly n points on an ellipsoid, using rejection sampling.

    Parameters
    ----------
    n : int, optional
        number of points, by default 2000
    a : int, optional
        ellipsoid param1, by default 3
    b : int, optional
        ellipsoid param2, by default 2
    c : int, optional
        stretchiness in z, by default 1
    seed : int, optional
        For repeatability, seed the randomness, by default None

    Returns
    -------
    The sampled points, and the curvatures of each point
    """

    np.random.seed(seed)
    theta, phi = rejection_sample_for_ellipsoid(n+1,a,b,c)
    theta = theta[1:]
    phi = phi[1:]
    data = np.zeros((len(theta), 3))
    data[:, 0] = a*np.cos(theta)* np.sin(phi)
    data[:, 1] = b*np.sin(theta)*np.sin(phi)
    data[:, 2] = c*np.cos(phi)

    # compute curvature of sampled torus (gaussian curvature)
    ks = 2* (a**2 * b**2 * c**2) / (a**2 * b**2 * np.cos(phi)**2 + c**2 * (b**2 * np.cos(theta)**2 + a**2 * np.sin(theta)**2)*np.sin(phi)**2)**2
    
    # add noise to data, if needed
    if noise:
        noise = np.random.normal(size = X.shape, loc = 0, scale = noise)
        X = X + noise
    
    return data, ks

# %% ../nbs/library/datasets/toy-datasets.ipynb 34
import sympy as sym
from .random_surfaces import rejection_sample_from_surface
def ellipsoid(n, noise=0, a=3, b=2, c=1, seed=None):
    np.random.seed(seed)
    theta = sym.Symbol("theta")
    phi = sym.Symbol("phi")
    f = sym.Matrix(
        [a*sym.cos(theta)*sym.sin(phi),b*sym.sin(theta)*sym.sin(phi),c*sym.cos(phi)]
    )
    X = rejection_sample_from_surface(f, n+1, bounds=[0,2*np.pi], seed=seed)[1:]
    # compute curvature of sampled torus (gaussian curvature)
    phi = np.arccos(X[:,2]/c)
    theta = np.arccos(X[:,0]*(1/a)*(1/np.sin(phi)))
    ks = 2* (a**2 * b**2 * c**2) / (a**2 * b**2 * np.cos(phi)**2 + c**2 * (b**2 * np.cos(theta)**2 + a**2 * np.sin(theta)**2)*np.sin(phi)**2)**2
    
    # add noise to data, if needed
    if noise:
        noise = np.random.normal(size = X.shape, loc = 0, scale = noise)
        X = X + noise
    return X, ks

# %% ../nbs/library/datasets/toy-datasets.ipynb 39
# import tadasets
import numpy as np
def sphere(n, d=2, radius = 1, use_guide_points = False, seed = None, noise = 0):
    np.random.seed(seed)
    if use_guide_points:
        n = n - 1
    X = np.random.randn(n,d+1)
    norm = np.linalg.norm(X,axis=1)
    X = X/norm[:,None]
    # u = np.random.normal(0,1,size=(n))
    # v = np.random.normal(0,1,size=(n))
    # w = np.random.normal(0,1,size=(n))
    # norm = (u*u + v*v + w*w)**(0.5)
    # (x,y,z) = (u,v,w)/norm
    # X = np.column_stack([x,y,z])
    # Use tadasets implementation
    # X = tadasets.dsphere(n, d=d)*radius
    ks = np.ones(n)*(2/radius**2)

    if noise:
        noise = np.random.normal(size = X.shape, loc = 0, scale = noise)
        X = X + noise
    # Compile guidepoints if needed
    if use_guide_points:
        X = np.vstack([np.array([0,0,1]),X])

    
    return X, ks

# %% ../nbs/library/datasets/toy-datasets.ipynb 46
def rejection_sample_for_saddle(n,a,b):
    x = np.random.random(n)*2 - 1 # random values in -1, 1
    y = np.random.random(n)*2 - 1
    fx = np.sqrt(4*a**2*x**2 + 4*b**2*y**2 + 1)
    yvec = np.random.random(n) * (1/np.max(fx))
    return x[yvec < fx], y[yvec < fx]

def paraboloid(n=2000,a=1,b=-1, seed=None, use_guide_points = False, noise = None):
    """Sample roughly n points on a saddle, using rejection sampling for even density coverage
    Defined by $ax^2 + by^2$. 

    Parameters
    ----------
    n : int, optional
        number of points, by default 2000
    a : int, optional
        ellipsoid param1, by default 1
    b : int, optional
        ellipsoid param2, by default -1
    seed : int, optional
        For repeatability, seed the randomness, by default None

    Returns
    -------
    The sampled points, and the curvatures of each point
    """
    if use_guide_points:
        n = n - 1
    np.random.seed(seed)
    x, y = rejection_sample_for_saddle(n,a,b)
    if use_guide_points:
        x = np.concatenate([[0],x])
        y = np.concatenate([[0],y])
    data = np.zeros((len(x), 3))
    data[:, 0] = x
    data[:, 1] = y
    data[:, 2] = a*x**2 + b*y**2
    # compute curvature of sampled saddle region
    # TODO: Compute gaussian curvature
    # TODO: Currently assuming that b is negative (hyperbolic paraboloid)
    ap = np.sqrt(1/a) 
    bp = b/np.abs(b) * np.sqrt(1/np.abs(b))
    ks = -(4*a**6 * b**6)/(a**4*b**4 + 4*b**4*x**2+4*a**4*y**2)**2

    X = data
    if noise:
        noise = np.random.normal(size = X.shape, loc = 0, scale = noise)
        X = X + noise
    
    return X, ks

# %% ../nbs/library/datasets/toy-datasets.ipynb 51
import sympy as sp
from .random_surfaces import rejection_sample_from_surface, scalar_curvature_at_origin
def rejection_sample_from_saddle(n_samples=1000, intrinsic_dim = 2, verbose=False, intensity=1):
    d = intrinsic_dim
    vars = sp.symbols('x0:%d' % d)
    saddle = sp.Matrix([*vars])
    for i in range(d,d+1):
        saddle = saddle.row_insert(i, sp.Matrix([intensity*sum([(-1)**j * vars[j]**2 for j in range(d)])]))
    if verbose: print(saddle)
    k = scalar_curvature_at_origin(saddle)
    return rejection_sample_from_surface(saddle, n_samples), k

# %% ../nbs/library/datasets/toy-datasets.ipynb 58
def plane(n, dim=2):
    coords_2d = np.random.rand(n-1,dim)*2-1
    coords_2d = np.vstack([np.zeros(dim),coords_2d])
    return coords_2d
