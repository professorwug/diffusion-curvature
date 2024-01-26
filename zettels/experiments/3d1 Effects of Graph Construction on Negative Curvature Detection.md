# 3d1 Effects of Graph Construction on Negative Curvature

::: {.cell 0=‘d’ 1=‘e’ 2=‘f’ 3=‘a’ 4=‘u’ 5=‘l’ 6=‘t’ 7=‘*’ 8=’e’ 9=’x’
10=’p’ 11=’ ’ 12=’3’ 13=’c’ 14=’1’ 15=’*’ 16=‘g’ 17=‘r’ 18=‘a’ 19=‘p’
20=‘h’ 21=‘*’ 22=’c’ 23=’o’ 24=’n’ 25=’s’ 26=’t’ 27=’r’ 28=’u’ 29=’c’
30=’t’ 31=’i’ 32=’o’ 33=’n’ 34=’*’ 35=‘e’ 36=‘x’ 37=‘p’ 38=‘e’ 39=‘r’
40=‘i’ 41=‘m’ 42=‘e’ 43=‘n’ 44=‘t’ 45=‘s’ 46=’ ’ 47=‘h’ 48=‘i’ 49=‘d’
50=‘e’}

``` python
## Standard libraries
import os
import math
import numpy as np
import time
# Configure environment
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false' # Tells Jax not to hog all of the memory to this process.

## Imports for plotting
import matplotlib.pyplot as plt
%matplotlib inline
from matplotlib.colors import to_rgba
import seaborn as sns
sns.set()

## Progress bar
from tqdm.notebook import tqdm, trange

## project specifics
import diffusion_curvature
from diffusion_curvature.datasets import *
from diffusion_curvature.graphs import *
from diffusion_curvature.core import *
import jax
import jax.numpy as jnp
print(jax.devices())

%load_ext autoreload
%autoreload 2
```

<div class="cell-output cell-output-stderr">

    An NVIDIA GPU may be present on this machine, but a CUDA-enabled jaxlib is not installed. Falling back to cpu.

</div>

<div class="cell-output cell-output-stdout">

    [CpuDevice(id=0)]

</div>

:::

In 3d, we noticed that the knn value has a suspiciously large effect on
the signs of curvature measured. With $k=5$, DC has trouble picking up
any negative curvature; with $k=15$, it struggles to identify anything
as positive.

Here we’ll test the method by probing its ability to separate saddles
and spheres in various dimensions.

## Executive Summary:

1.  Both the fixed and adaptive kernel significantly outperform
    Graphtools’ graph construction on this toy data.
2.  The median heuristic is not very good at choosing the right sigma.
3.  The adaptive gaussian kernel with $k=10$ chooses the right sigma
    well, producing tightly clustered distinctions between the saddles
    and spheres.
4.  Disabling the anisotropic density normalization *helps* performance
    in high dimensions, enabling the adaptive kernel + DC to detect
    negative curvature in higher-dimensional saddles.

# With Graphtools’ Graphs

``` python
import graphtools
from diffusion_curvature.core import DiffusionCurvature
from diffusion_curvature.datasets import rejection_sample_from_saddle, sphere
from functools import partial

def graphtools_graph_from_data(X, knn = 15):
    return graphtools.Graph(X, anisotropy=1, knn=knn, decay=None).to_pygsp()
    
def get_dc_of_saddles_and_spheres(
    dim = 3,
    num_samplings = 100,
    graph_former = graphtools_graph_from_data,
    return_data = False,
    t = 25,
    title = "",
    smoothing = 1,
):
    samplings = [2000]*num_samplings
    ks_dc_saddles = []
    ks_dc_spheres = []
    X_saddles_sampled = []
    X_spheres_sampled = []
    
    for n_points in tqdm(samplings):
        X_saddle, k_saddle = rejection_sample_from_saddle(n_points, dim)
        X_saddles_sampled.append(X_saddle)
        X_sphere, k_sphere = sphere(n_points, d=dim)
        X_spheres_sampled.append(X_sphere)
        # Compute Diffusion Curvature on Sphere
        G = graph_former(X_sphere)
        DC = DiffusionCurvature(
            laziness_method="Entropic",
            flattening_method="Fixed",
            comparison_method="Subtraction",
            graph_former = graph_former,
            points_per_cluster=None, # construct separate comparison spaces around each point
            comparison_space_size_factor=1,
            smoothing = smoothing,
            verbose = True,
        )
        ks = DC.curvature(G, t=t, dim=dim, idx=0)
        ks_dc_spheres.append(ks)
        # Compute Diffusion Curvature on Saddle
        G = graph_former(X_saddle)
        DC = DiffusionCurvature(
            laziness_method="Entropic",
            flattening_method="Fixed",
            comparison_method="Subtraction",
            graph_former = graph_former,
            points_per_cluster=None, # construct separate comparison spaces around each point
            comparison_space_size_factor=1,
            smoothing = smoothing,
        )
        ks = DC.curvature(G, t=t, dim=dim, idx=0)
        ks_dc_saddles.append(ks)
    
    # plot a histogram of the diffusion curvatures
    plt.hist(ks_dc_saddles, bins=50, color='orange', label = 'Saddles')
    plt.hist(ks_dc_spheres, bins=50, color='green', label = 'Spheres')
    plt.legend()
    plt.xlabel("Diffusion Curvature")
    plt.title(f"{title} In dimension {dim} with {t=}")
    # plt.xlabel('')
    # plt.ylabel('Frequency')
    if return_data: return ks_dc_saddles, ks_dc_spheres
```

``` python
get_dc_of_saddles_and_spheres(
    graph_former = partial(graphtools_graph_from_data, knn=30)
)
```

      0%|          | 0/100 [00:00<?, ?it/s]

    Manifold spreads are 7.429889678955078
    average_dim_in_cluster=3
    comparison entropy is 7.571398735046387
    Manifold spreads are 7.4370527267456055
    average_dim_in_cluster=3
    comparison entropy is 7.566686630249023
    Manifold spreads are 7.439205646514893
    average_dim_in_cluster=3
    comparison entropy is 7.579890251159668
    Manifold spreads are 7.428216934204102
    average_dim_in_cluster=3
    comparison entropy is 7.5763397216796875
    Manifold spreads are 7.4570512771606445
    average_dim_in_cluster=3
    comparison entropy is 7.579254627227783
    Manifold spreads are 7.433935165405273
    average_dim_in_cluster=3
    comparison entropy is 7.557219505310059
    Manifold spreads are 7.433426380157471
    average_dim_in_cluster=3
    comparison entropy is 7.575214385986328
    Manifold spreads are 7.4390549659729
    average_dim_in_cluster=3
    comparison entropy is 7.5724968910217285
    Manifold spreads are 7.425068378448486
    average_dim_in_cluster=3
    comparison entropy is 7.570180892944336
    Manifold spreads are 7.450780391693115
    average_dim_in_cluster=3
    comparison entropy is 7.587141036987305
    Manifold spreads are 7.4259490966796875
    average_dim_in_cluster=3
    comparison entropy is 7.571746826171875
    Manifold spreads are 7.440629005432129
    average_dim_in_cluster=3
    comparison entropy is 7.573746681213379
    Manifold spreads are 7.450885772705078
    average_dim_in_cluster=3
    comparison entropy is 7.57297945022583
    Manifold spreads are 7.455705642700195
    average_dim_in_cluster=3
    comparison entropy is 7.567692756652832
    Manifold spreads are 7.428769588470459
    average_dim_in_cluster=3
    comparison entropy is 7.573057174682617
    Manifold spreads are 7.4542236328125
    average_dim_in_cluster=3
    comparison entropy is 7.582261562347412
    Manifold spreads are 7.428750991821289
    average_dim_in_cluster=3
    comparison entropy is 7.573791027069092

    KeyboardInterrupt: 

``` python
get_dc_of_saddles_and_spheres(
    graph_former = partial(graphtools_graph_from_data, knn=20)
)
```

      0%|          | 0/100 [00:00<?, ?it/s]

![](3d1%20Effects%20of%20Graph%20Construction%20on%20Negative%20Curvature%20Detection_files/figure-commonmark/cell-5-output-2.png)

``` python
get_dc_of_saddles_and_spheres(
    graph_former = partial(graphtools_graph_from_data, knn=15),
    dim=2
)
```

      0%|          | 0/100 [00:00<?, ?it/s]

![](3d1%20Effects%20of%20Graph%20Construction%20on%20Negative%20Curvature%20Detection_files/figure-commonmark/cell-6-output-2.png)

``` python
get_dc_of_saddles_and_spheres(
    graph_former = partial(graphtools_graph_from_data, knn=15),
    dim=3
)
```

      0%|          | 0/100 [00:00<?, ?it/s]

![](3d1%20Effects%20of%20Graph%20Construction%20on%20Negative%20Curvature%20Detection_files/figure-commonmark/cell-7-output-2.png)

``` python
get_dc_of_saddles_and_spheres(
    graph_former = partial(graphtools_graph_from_data, knn=15),
    dim=4,
)
```

      0%|          | 0/100 [00:00<?, ?it/s]

![](3d1%20Effects%20of%20Graph%20Construction%20on%20Negative%20Curvature%20Detection_files/figure-commonmark/cell-8-output-2.png)

``` python
get_dc_of_saddles_and_spheres(
    graph_former = partial(graphtools_graph_from_data, knn=15),
    dim=5,
)
```

      0%|          | 0/100 [00:00<?, ?it/s]

![](3d1%20Effects%20of%20Graph%20Construction%20on%20Negative%20Curvature%20Detection_files/figure-commonmark/cell-9-output-2.png)

``` python
get_dc_of_saddles_and_spheres(
    graph_former = partial(graphtools_graph_from_data, knn=10)
)
```

      0%|          | 0/100 [00:00<?, ?it/s]

![](3d1%20Effects%20of%20Graph%20Construction%20on%20Negative%20Curvature%20Detection_files/figure-commonmark/cell-10-output-2.png)

``` python
get_dc_of_saddles_and_spheres(
    graph_former = partial(graphtools_graph_from_data, knn=5)
)
```

      0%|          | 0/100 [00:00<?, ?it/s]

![](3d1%20Effects%20of%20Graph%20Construction%20on%20Negative%20Curvature%20Detection_files/figure-commonmark/cell-11-output-2.png)

``` python
get_dc_of_saddles_and_spheres(
    graph_former = partial(graphtools_graph_from_data, knn=15),
    dim = 2
)
```

      0%|          | 0/100 [00:00<?, ?it/s]

![](3d1%20Effects%20of%20Graph%20Construction%20on%20Negative%20Curvature%20Detection_files/figure-commonmark/cell-12-output-2.png)

``` python
get_dc_of_saddles_and_spheres(
    graph_former = partial(graphtools_graph_from_data, knn=10),
    dim = 2
)
```

      0%|          | 0/100 [00:00<?, ?it/s]

![](3d1%20Effects%20of%20Graph%20Construction%20on%20Negative%20Curvature%20Detection_files/figure-commonmark/cell-13-output-2.png)

``` python
get_dc_of_saddles_and_spheres(
    graph_former = partial(graphtools_graph_from_data, knn=5),
    dim = 2
)
```

      0%|          | 0/100 [00:00<?, ?it/s]

![](3d1%20Effects%20of%20Graph%20Construction%20on%20Negative%20Curvature%20Detection_files/figure-commonmark/cell-14-output-2.png)

# With a Plain Gaussian Kernel

We observe that the variance between samplings is drastically reduced
when using a plain gaussian kernel.

``` python
from diffusion_curvature.kernels import gaussian_kernel
from dataclasses import dataclass

@dataclass
class SimpleGraph:
    W: np.ndarray
    
def graph_from_gaussian_kernel(X, sigma = 0, alpha = 0):
    W = gaussian_kernel(
        X,
        kernel_type='fixed',
        sigma = sigma, # use median heuristic
        anisotropic_density_normalization = alpha,
    )
    G = SimpleGraph(W = W)
    return G
```

``` python
X, ks = sphere(1000)
G = graph_from_gaussian_kernel(X,sigma=0.7)
# assert np.allclose(G.W,gaussian_kernel(X, kernel_type='fixed',sigma=0, anisotropic_density_normalization=1), atol = 1e-9)
```

``` python
G.W
```

    array([[0.56991754, 0.14405797, 0.0124326 , ..., 0.01224254, 0.36494885,
            0.25089434],
           [0.14405797, 0.56991754, 0.0835226 , ..., 0.02234097, 0.4144598 ,
            0.34591576],
           [0.0124326 , 0.0835226 , 0.56991754, ..., 0.35478198, 0.02973049,
            0.02779005],
           ...,
           [0.01224254, 0.02234097, 0.35478198, ..., 0.56991754, 0.01237804,
            0.01171373],
           [0.36494885, 0.4144598 , 0.02973049, ..., 0.01237804, 0.56991754,
            0.39543834],
           [0.25089434, 0.34591576, 0.02779005, ..., 0.01171373, 0.39543834,
            0.56991754]])

``` python
D = np.diag(1/np.sum(G.W, axis=1))
```

``` python
np.max(G.W)
```

    0.5699175434306182

``` python
from sklearn.metrics import pairwise_distances
from diffusion_curvature.kernels import median_heuristic
```

``` python
D = pairwise_distances(X)
median_heuristic(D)
```

    0.9999075219177865

``` python
ksaddles, kspheres = get_dc_of_saddles_and_spheres(
    graph_former = partial(graph_from_gaussian_kernel, sigma=2),
    dim = 5,
    return_data = True
)
```

      0%|          | 0/100 [00:00<?, ?it/s]

![](3d1%20Effects%20of%20Graph%20Construction%20on%20Negative%20Curvature%20Detection_files/figure-commonmark/cell-22-output-2.png)

``` python
ksaddles, kspheres = get_dc_of_saddles_and_spheres(
    graph_former = partial(graph_from_gaussian_kernel, sigma=1),
    dim = 2,
    return_data = True,
    t=25,
)
```

      0%|          | 0/100 [00:00<?, ?it/s]

![](3d1%20Effects%20of%20Graph%20Construction%20on%20Negative%20Curvature%20Detection_files/figure-commonmark/cell-23-output-2.png)

``` python
plt.hist(kspheres)
```

    (array([ 1.,  2.,  1.,  8., 14., 25., 18., 22.,  7.,  2.]),
     array([-0.02466249, -0.02416816, -0.02367382, -0.02317948, -0.02268515,
            -0.02219081, -0.02169647, -0.02120214, -0.0207078 , -0.02021346,
            -0.01971912]),
     <BarContainer object of 10 artists>)

![](3d1%20Effects%20of%20Graph%20Construction%20on%20Negative%20Curvature%20Detection_files/figure-commonmark/cell-24-output-2.png)

``` python
plt.hist(ksaddles)
```

    (array([ 4.,  5., 11., 19., 19., 27.,  7.,  5.,  2.,  1.]),
     array([0.00434542, 0.00509653, 0.00584764, 0.00659876, 0.00734987,
            0.00810099, 0.0088521 , 0.00960321, 0.01035433, 0.01110544,
            0.01185656]),
     <BarContainer object of 10 artists>)

![](3d1%20Effects%20of%20Graph%20Construction%20on%20Negative%20Curvature%20Detection_files/figure-commonmark/cell-25-output-2.png)

Aha! Turns out 0.1 is the correct sigma value for this problem.
Evidently the median heuristic is terrible for this use case.

Interestingly, with the right sigma parameter, the plain gaussian kernel
separates spheres and saddles far more effectively than graphtools.

``` python
ksaddles, kspheres = get_dc_of_saddles_and_spheres(
    graph_former = partial(
        graph_from_gaussian_kernel, sigma=2, alpha=0),
    dim = 2, 
    return_data = True
)
```

      0%|          | 0/100 [00:00<?, ?it/s]

![](3d1%20Effects%20of%20Graph%20Construction%20on%20Negative%20Curvature%20Detection_files/figure-commonmark/cell-26-output-2.png)

``` python
plt.hist(kspheres)
```

    (array([ 2.,  2.,  4., 12., 28., 19., 10., 12.,  8.,  3.]),
     array([0.96014023, 1.00001037, 1.03988051, 1.07975078, 1.11962092,
            1.15949106, 1.19936121, 1.23923135, 1.27910161, 1.31897175,
            1.3588419 ]),
     <BarContainer object of 10 artists>)

![](3d1%20Effects%20of%20Graph%20Construction%20on%20Negative%20Curvature%20Detection_files/figure-commonmark/cell-27-output-2.png)

What happens with anisotropy now that we have the right sigma? I expect
it to decrease the variance. It does, somewhat - but it also distorts
the numbers. It would seem that negative curvature is somehow adversely
affected by complete anisotropic normalization.

``` python
plt.hist(ksaddles)
```

    (array([ 3.,  5., 10., 14., 15., 10., 20.,  5., 12.,  6.]),
     array([0.20184517, 0.22028217, 0.23871918, 0.25715619, 0.27559319,
            0.29403019, 0.31246719, 0.33090419, 0.34934121, 0.36777821,
            0.38621521]),
     <BarContainer object of 10 artists>)

![](3d1%20Effects%20of%20Graph%20Construction%20on%20Negative%20Curvature%20Detection_files/figure-commonmark/cell-28-output-2.png)

``` python
ksaddles, kspheres = get_dc_of_saddles_and_spheres(
    graph_former = partial(
        graph_from_gaussian_kernel, sigma=0.1, alpha=1),
    dim = 2, 
    return_data = True
)
```

      0%|          | 0/100 [00:00<?, ?it/s]

![](3d1%20Effects%20of%20Graph%20Construction%20on%20Negative%20Curvature%20Detection_files/figure-commonmark/cell-29-output-2.png)

``` python
plt.hist(kspheres)
```

    (array([ 1.,  2.,  8., 18., 19., 17., 19.,  8.,  6.,  2.]),
     array([0.90808916, 0.93991822, 0.97174728, 1.0035764 , 1.0354054 ,
            1.06723452, 1.09906363, 1.13089263, 1.16272175, 1.19455075,
            1.22637987]),
     <BarContainer object of 10 artists>)

![](3d1%20Effects%20of%20Graph%20Construction%20on%20Negative%20Curvature%20Detection_files/figure-commonmark/cell-30-output-2.png)

``` python
plt.hist(ksaddles)
```

    (array([ 2.,  4., 14., 19., 17., 20., 13.,  7.,  3.,  1.]),
     array([0.01172924, 0.02543383, 0.03913841, 0.052843  , 0.06654759,
            0.08025217, 0.09395675, 0.10766134, 0.12136593, 0.13507052,
            0.1487751 ]),
     <BarContainer object of 10 artists>)

![](3d1%20Effects%20of%20Graph%20Construction%20on%20Negative%20Curvature%20Detection_files/figure-commonmark/cell-31-output-2.png)

# The Adaptive Kernel

I thought this was what graphtools was doing; evidently, I was woefully
wrong. There’s far less variance, and far more accurate readings here
than with graphtools.

``` python
from diffusion_curvature.kernels import gaussian_kernel
from dataclasses import dataclass
```

``` python
def graph_from_adaptive_gaussian_kernel(X, k=10, alpha = 1):
    W = gaussian_kernel(
        X,
        kernel_type='adaptive',
        k = k,
        anisotropic_density_normalization = alpha,
    )
    G = SimpleGraph(W = W)
    return G
```

``` python
ksaddles, kspheres = get_dc_of_saddles_and_spheres(
    graph_former = partial(
        graph_from_adaptive_gaussian_kernel, k=10, alpha=1),
    dim = 2, 
    return_data = True
)
```

      0%|          | 0/100 [00:00<?, ?it/s]

    KeyboardInterrupt: 

``` python
ksaddles, kspheres = get_dc_of_saddles_and_spheres(
    graph_former = partial(
        graph_from_adaptive_gaussian_kernel, k=5, alpha=1),
    dim = 2, 
    return_data = True
)
```

      0%|          | 0/100 [00:00<?, ?it/s]

![](3d1%20Effects%20of%20Graph%20Construction%20on%20Negative%20Curvature%20Detection_files/figure-commonmark/cell-35-output-2.png)

``` python
ksaddles, kspheres = get_dc_of_saddles_and_spheres(
    graph_former = partial(
        graph_from_adaptive_gaussian_kernel, k=10, alpha=1),
    dim = 3, 
    return_data = True
)
```

      0%|          | 0/100 [00:00<?, ?it/s]

![](3d1%20Effects%20of%20Graph%20Construction%20on%20Negative%20Curvature%20Detection_files/figure-commonmark/cell-36-output-2.png)

``` python
ksaddles, kspheres = get_dc_of_saddles_and_spheres(
    graph_former = partial(
        graph_from_adaptive_gaussian_kernel, k=5, alpha=1),
    dim = 3, 
    return_data = True
)
```

      0%|          | 0/100 [00:00<?, ?it/s]

![](3d1%20Effects%20of%20Graph%20Construction%20on%20Negative%20Curvature%20Detection_files/figure-commonmark/cell-37-output-2.png)

So it appears that my adaptive kernel function does a much better job of
distinguishing between positive and negative. However, there is still
some slippage in high dimensions. Where saddles are seen as positively
curved. I suspect this relates to the problem of holes in the data.

``` python
ksaddles, kspheres = get_dc_of_saddles_and_spheres(
    graph_former = partial(
        graph_from_adaptive_gaussian_kernel, k=10, alpha=0),
    dim = 2, 
    return_data = True,
    t=10
)
```

      0%|          | 0/100 [00:00<?, ?it/s]

![](3d1%20Effects%20of%20Graph%20Construction%20on%20Negative%20Curvature%20Detection_files/figure-commonmark/cell-38-output-2.png)

``` python
ksaddles, kspheres = get_dc_of_saddles_and_spheres(
    graph_former = partial(
        graph_from_adaptive_gaussian_kernel, k=10, alpha=0),
    dim = 3, 
    return_data = True
)
```

      0%|          | 0/100 [00:00<?, ?it/s]

![](3d1%20Effects%20of%20Graph%20Construction%20on%20Negative%20Curvature%20Detection_files/figure-commonmark/cell-39-output-2.png)

``` python
ksaddles, kspheres = get_dc_of_saddles_and_spheres(
    graph_former = partial(
        graph_from_adaptive_gaussian_kernel, k=10, alpha=0),
    dim = 3, 
    return_data = True
)
```

      0%|          | 0/100 [00:00<?, ?it/s]

![](3d1%20Effects%20of%20Graph%20Construction%20on%20Negative%20Curvature%20Detection_files/figure-commonmark/cell-40-output-2.png)

How bizarre, removing the anisotropic density correction, restores some
of the detection of negative curvature. Why could this be? Perhaps it’s
because positive and negatively curved areas actually have different
densities. By normalizing this, we’re forcing them to look more alike.

``` python
ksaddles, kspheres = get_dc_of_saddles_and_spheres(
    graph_former = partial(
        graph_from_adaptive_gaussian_kernel, k=10, alpha=0),
    dim = 4, 
    return_data = True
)
```

      0%|          | 0/100 [00:00<?, ?it/s]

![](3d1%20Effects%20of%20Graph%20Construction%20on%20Negative%20Curvature%20Detection_files/figure-commonmark/cell-41-output-2.png)

``` python
ksaddles, kspheres = get_dc_of_saddles_and_spheres(
    graph_former = partial(
        graph_from_adaptive_gaussian_kernel, k=10, alpha=0),
    dim = 5, 
    return_data = True
)
```

      0%|          | 0/100 [00:00<?, ?it/s]

![](3d1%20Effects%20of%20Graph%20Construction%20on%20Negative%20Curvature%20Detection_files/figure-commonmark/cell-42-output-2.png)

# Intrinsic vs Extrinsic Curvature Measurements

I suspect there are two factors captured by diffusion curvature: an
intrinsic laziness-based curvature, and an extrinsic kernel-based
curvature.

1.  *The laziness of multi-step random walk*s, building off the fact
    that in positive curvature ‘the neighbors of neighbors have more
    neighbors in common’. This emerges when the kernel bandwidth is low
    and $t$ is high. This is what we captured with the original
    diffusion laziness formulation.
2.  *The distribution of points in ambient space as captured by a
    spherical kernel*. In positive curvature settings, the points are
    concentrated near the center of the kernel. In negative curvature
    settings, they’re spread out. Hence the *shape* of even a single
    step of diffusion captures this *extrinsic* measure.

The above experiments suggest that the extrinsic approach more reliably
distinguishes saddles and spheres. Let’s see if we can tune the
intrinsic approach to perform comparably.

First, here’s the extrinsic method at the height of its powers. The
kernel is *huge*.

``` python
ksaddles, kspheres = get_dc_of_saddles_and_spheres(
    graph_former = partial(graph_from_gaussian_kernel, sigma=2),
    dim = 2,
    return_data = True,
    title = "Extrinsic kernel-based:"
)
```

      0%|          | 0/100 [00:00<?, ?it/s]

![](3d1%20Effects%20of%20Graph%20Construction%20on%20Negative%20Curvature%20Detection_files/figure-commonmark/cell-43-output-2.png)

Now let’s try the intrinsic, laziness-based method. We’ll use
traditional diffusion laziness, with an aperture of 20, and one
application of smoothing.

``` python
import graphtools
from diffusion_curvature.core import DiffusionCurvature
from diffusion_curvature.datasets import rejection_sample_from_saddle, sphere
from functools import partial

def get_dc_of_saddles_and_spheres(
    dim = 3,
    num_samplings = 100,
    graph_former = graphtools_graph_from_data,
    return_data = False,
    t = 25,
    title = "",
    smoothing = 1,
):
    samplings = [2000]*num_samplings
    ks_dc_saddles = []
    ks_dc_spheres = []
    X_saddles_sampled = []
    X_spheres_sampled = []
    
    for n_points in tqdm(samplings):
        X_saddle, k_saddle = rejection_sample_from_saddle(n_points, dim)
        X_saddles_sampled.append(X_saddle)
        X_sphere, k_sphere = sphere(n_points, d=dim)
        X_spheres_sampled.append(X_sphere)
        # Compute Diffusion Curvature on Sphere
        G = graph_former(X_sphere)
        DC = DiffusionCurvature(
            laziness_method="Laziness",
            flattening_method="Fixed",
            comparison_method="Subtraction",
            graph_former = graph_former,
            points_per_cluster=None, # construct separate comparison spaces around each point
            comparison_space_size_factor=1,
            smoothing = smoothing,
        )
        ks = DC.curvature(G, t=t, dim=dim, idx=0)
        ks_dc_spheres.append(ks)
        # Compute Diffusion Curvature on Saddle
        G = graph_former(X_saddle)
        DC = DiffusionCurvature(
            laziness_method="Laziness",
            flattening_method="Fixed",
            comparison_method="Subtraction",
            graph_former = graph_former,
            points_per_cluster=None, # construct separate comparison spaces around each point
            comparison_space_size_factor=1,
            smoothing = smoothing,
        )
        ks = DC.curvature(G, t=t, dim=dim, idx=0)
        ks_dc_saddles.append(ks)
    
    # plot a histogram of the diffusion curvatures
    plt.hist(ks_dc_saddles, bins=50, color='orange', label = 'Saddles')
    plt.hist(ks_dc_spheres, bins=50, color='green', label = 'Spheres')
    plt.legend()
    plt.xlabel("Diffusion Curvature")
    plt.title(f"{title} In dimension {dim} with {t=}")
    # plt.xlabel('')
    # plt.ylabel('Frequency')
    if return_data: return ks_dc_saddles, ks_dc_spheres
```

``` python
ksaddles, kspheres = get_dc_of_saddles_and_spheres(
    graph_former = partial(graph_from_gaussian_kernel, sigma=0.1, alpha=1), # use high anisotropy to normalize out density
    dim = 2,
    return_data = True,
    t=25,
)
```

      0%|          | 0/100 [00:00<?, ?it/s]

![](3d1%20Effects%20of%20Graph%20Construction%20on%20Negative%20Curvature%20Detection_files/figure-commonmark/cell-45-output-2.png)

So, with the laziness method, a bit of smoothing, and a high t value,
the intrinsic method can reliably distinguish between negative and
positive curvature.

``` python
ksaddles, kspheres = get_dc_of_saddles_and_spheres(
    graph_former = partial(graph_from_gaussian_kernel, sigma=0.1, alpha=1), # use high anisotropy to normalize out density
    dim = 3,
    return_data = True,
    t=25,
)
```

      0%|          | 0/100 [00:00<?, ?it/s]

![](3d1%20Effects%20of%20Graph%20Construction%20on%20Negative%20Curvature%20Detection_files/figure-commonmark/cell-46-output-2.png)

``` python
ksaddles, kspheres = get_dc_of_saddles_and_spheres(
    graph_former = partial(graph_from_gaussian_kernel, sigma=0.1, alpha=1), # use high anisotropy to normalize out density
    dim = 4,
    return_data = True,
    t=25,
)
```

      0%|          | 0/100 [00:00<?, ?it/s]

![](3d1%20Effects%20of%20Graph%20Construction%20on%20Negative%20Curvature%20Detection_files/figure-commonmark/cell-47-output-2.png)

``` python
ksaddles, kspheres = get_dc_of_saddles_and_spheres(
    graph_former = partial(graph_from_gaussian_kernel, sigma=0.1, alpha=1), # use high anisotropy to normalize out density
    dim = 4,
    return_data = True,
    t=40,
)
```

      0%|          | 0/100 [00:00<?, ?it/s]

![](3d1%20Effects%20of%20Graph%20Construction%20on%20Negative%20Curvature%20Detection_files/figure-commonmark/cell-48-output-2.png)

# Repeating with Sparser Graphs
