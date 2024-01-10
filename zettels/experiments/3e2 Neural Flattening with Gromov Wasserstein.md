# 3e2 Neural Flattening with Gromov Wasserstein

<!-- WARNING: THIS FILE WAS AUTOGENERATED! DO NOT EDIT! -->

Any measurement of curvature must deal with the confounding influence of
density. The high-frequency effects of sampling, including regions of
sparsity and pockets of density, are locally indistinguishable from
curvature. There are two ways of dealing with this: we can either build
a sufficiently robust measurement with high enough noise tolerance that
these local density effects don’t matter, or we can adapt the
measurement to them, by comparing to a euclidean space with
approximately the same sampling.

This notebook explores one move towards the second option, the Gromov
Wasserstein distance, which takes an OT like distance between points in
distinct space. It works by searching over all possible couplings of
points between the two spaces $X,Y$ to minimize the discrepancy within a
coupling $\gamma$, defined as

$$\int \int d_{X}(x - x') - d_{Y}(y - y') \, d\gamma(x,y) \, d \gamma(x',y')$$

The Gromov Wasserstein distance is routinely used to recognize objects
that might be changing shape, for example several pointclouds made from
a hand with fingers in positions of varying curvature. This suggests
that GW gives us some robustness against curvature, being partially
blind to it. As long as the points within the manifold have the same
proximal relation to each other, their orientation in space doesn’t
matter.

So here’s the idea: given a manifold region surrounding some point whose
curvature we want to know, we’ll use GW to construct a *neurally
flattened* version of this region by learning a positioning of points in
a euclidean space of the manifold’s intrinsic dimension to minimize the
GW distance between this flattened space and the manifold. Ideally, this
will recapitulate all of the density artifacts of our region, but now
within a flattened space.

# How does GW reconstruct curvature vs density?

The trouble, of course, is that curvature - over a global level - *does*
change the relations of points. In higher curvature settings, there’s
more interconnectivity. **Our goal with this preliminary experiment is
to assess the degree to which the curvature-based differences in graph
connectivity are weighted by GW versus the density-based differences.**
The hypothesis is that density artifacts are preserved more faithfully.

To measure this, we’ll rig together a simple test case by measuring the
GW distance between

1.  a neighborhood sampled from a sphere
2.  PCA projections of this neighborhood, with various amounts of noise
    added

We’ll then visually inspect the PCA projections with the lowest
distance, to see how well they recreate the sampling artifacts on the
sphere.

``` python
X_sphere, ks_sphere = sphere(1000) # keep it relatively sparse
X_cap_of_sphere = X_sphere[X_sphere[:,2] > 0.7] # Just the itty bitty polar top
```

``` python
print(len(X_cap_of_sphere))
plot_3d(X_cap_of_sphere,use_plotly=True)
plt.scatter(X_cap_of_sphere[:,0],X_cap_of_sphere[:,1])
```

    143

        <script type="text/javascript">
        window.PlotlyConfig = {MathJaxConfig: 'local'};
        if (window.MathJax && window.MathJax.Hub && window.MathJax.Hub.Config) {window.MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}
        if (typeof require !== 'undefined') {
        require.undef("plotly");
        define('plotly', function(require, exports, module) {
            /**
* plotly.js v2.27.0
* Copyright 2012-2023, Plotly, Inc.
* All rights reserved.
* Licensed under the MIT license
*/
/*! For license information please see plotly.min.js.LICENSE.txt */
        });
        require(['plotly'], function(Plotly) {
            window._Plotly = Plotly;
        });
        }
        </script>
        &#10;
![](3e2%20Neural%20Flattening%20with%20Gromov%20Wasserstein_files/figure-commonmark/cell-3-output-3.png)

    <matplotlib.collections.PathCollection at 0x7f85326081d0>

![](3e2%20Neural%20Flattening%20with%20Gromov%20Wasserstein_files/figure-commonmark/cell-3-output-5.png)

The PCA projection here is just the x and y coords. Note that although
this is ‘flattened’, it retains some peculiarities of the projection.
For instance, the density of points is higher around the edges and
sparser in the middle.

``` python
X_pca = X_cap_of_sphere[:,:2]
```

We hope the Gromov-Wasserstein will reward flat spaces that retain the
density quirks but get rid of the these projection artifacts. To
measure, we’ll sample several flat spaces

1.  Different samples of noise added to `X_pca`
2.  Random samplings of the plane.

``` python
from diffusion_curvature.core import get_adaptive_graph
```

``` python
A_pca = get_adaptive_graph(X_pca).W
plt.imshow(A_pca)
```

    <matplotlib.image.AxesImage at 0x7f84007174d0>

![](3e2%20Neural%20Flattening%20with%20Gromov%20Wasserstein_files/figure-commonmark/cell-6-output-2.png)

``` python
import ot
```

``` python
class NoisedPCA():
    def __init__(self, X, sigma):
        self.sigma = sigma
        self.noise = np.random.normal(size = X.shape, loc = 0, scale = sigma)
        self.X = X + self.noise
        self.A = get_adaptive_graph(self.X).W
        self.A /= self.A.max()
    def compute_distance(self, A_real):
        # Projected Gradient algorithm with entropic regularization
        A_real /= A_real.max()
        n_samples = len(A_real)
        p = ot.unif(n_samples)
        q = ot.unif(n_samples)
        gwe, loge = ot.gromov.entropic_gromov_wasserstein(
        self.A, A_real, p, q, 'square_loss', epsilon=5e-4, solver='PGD',
        log=True, verbose=False)
        self.distance = loge['gw_dist']
        self.coupling = gwe # the coupling, T

comparison_spaces = []
for sigma in tqdm([0.05, 0.1, 0.15, 0.2, 0.3]):
    for trials in trange(10, leave=False):
        NP = NoisedPCA(X_pca, sigma)
        NP.compute_distance(A_pca)
        comparison_spaces.append(NP)
```

      0%|          | 0/5 [00:00<?, ?it/s]

      0%|          | 0/10 [00:00<?, ?it/s]

      0%|          | 0/10 [00:00<?, ?it/s]

      0%|          | 0/10 [00:00<?, ?it/s]

      0%|          | 0/10 [00:00<?, ?it/s]

      0%|          | 0/10 [00:00<?, ?it/s]

``` python
comparison_spaces.sort(key=lambda x: x.distance)
```

``` python
# order the spaces with corresponding idxs, using the coupling map
def order_comparison_space(C):
    ordering = np.arange(len(C.T)) @ C.T
    X_ordered = C.X[ordering]
```

``` python
i = 1
plt.scatter(comparison_spaces[i].X[:,0], comparison_spaces[i].X[:,1])
plt.title(f"Sampling {i} with sigma {comparison_spaces[i].sigma} has GW Distance {comparison_spaces[i].distance}")
```

    Text(0.5, 1.0, 'Sampling 1 with sigma 0.05 has GW Distance 0.0022035801608677992')

![](3e2%20Neural%20Flattening%20with%20Gromov%20Wasserstein_files/figure-commonmark/cell-11-output-2.png)

``` python
i = 5
plt.scatter(comparison_spaces[i].X[:,0], comparison_spaces[i].X[:,1])
plt.title(f"Sampling {i} with sigma {comparison_spaces[i].sigma} has GW Distance {comparison_spaces[i].distance}")
```

    Text(0.5, 1.0, 'Sampling 5 with sigma 0.05 has GW Distance 0.0026334806113825094')

![](3e2%20Neural%20Flattening%20with%20Gromov%20Wasserstein_files/figure-commonmark/cell-12-output-2.png)