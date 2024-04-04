# AUTOGENERATED! DO NOT EDIT! File to edit: ../../../nbs/experiments/2c3-are-kernels-zeitgeibers.ipynb.

# %% auto 0
__all__ = ['show_curvature_curves']

# %% ../../../nbs/experiments/2c3-are-kernels-zeitgeibers.ipynb 6
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from fastcore.all import *
from ..core import *
from tqdm.auto import tqdm
import inspect

@patch
def curvature_curve(self:DiffusionCurvature, 
                    num_ts=50, 
                    idx = 0, 
                    curvature_fn:str = 'unsigned_curvature', # name of local DiffusionCurvature method to call. Should take as input G and t and kwargs
                    **kwargs
                    ):
    method = getattr(self, curvature_fn, lambda: "Function not found")
    
    t_values = np.arange(1, num_ts+1)
    curvatures = np.array([method(self.G, int(t), **kwargs)[idx] for t in tqdm(t_values)])
    return t_values, curvatures

def show_curvature_curves(*diffusion_curvatures, num_ts = 50, idx = 0, scaling_fn = None, title = "Curvature Curves", curvature_fn = "unsigned_curvature", **kwargs):
    # fig, ax = plt.subplots(1)
    for dc in diffusion_curvatures:
        for frame_record in inspect.stack():
            frame = frame_record.frame
            for name, obj in frame.f_globals.items():
                if obj is dc:
                    dc_name = name
            for name, obj in frame.f_locals.items():
                if obj is dc:
                    dc_name = name
        t_values, curvatures = dc.curvature_curve(num_ts = num_ts, idx=idx, curvature_fn=curvature_fn, **kwargs)
        plt.plot(t_values, curvatures, label=dc_name)
    plt.xlabel('Time ($t$)')
    plt.ylabel('Curvature')
    plt.title(title)
    plt.legend()
    plt.show()
