# Measurements of Diffusion Laziness

``` python
# Diffusion Curvature utils
from diffusion_curvature.kernels import *
from diffusion_curvature.datasets import *
from diffusion_curvature.core import *
from diffusion_curvature.datasets import *
from diffusion_curvature.kernels import *
# Python necessities
import numpy as np
from fastcore.all import *
import matplotlib.pyplot as plt
# Notebook Helpers
from nbdev.showdoc import *
from tqdm.notebook import trange, tqdm
%load_ext autoreload
%autoreload 2
```

> Quantifies how much each diffusion spreads. This value alone gives
> curvature magnitude; when a comparison space is included, it becomes
> signed.

# Entropic Diffusion Laziness

``` python
import graphtools
from scipy.stats import entropy

def entropy_of_diffusion(G:graphtools.graphs.DataGraph, idx=None):
        """
        Returns the pointwise entropy of diffusion from the powered diffusion matrix in the inpiut 
        """
        assert G.Pt is not None
        # TODO: Entropy may be compatible with scipy.sparse matrices
        if type(G.Pt) == np.ndarray:
            Pt = G.Pt
        else:
            Pt = G.Pt.toarray()
        if idx is None:
            return entropy(Pt, axis=1)
        else:
            return entropy(Pt[idx])
```
