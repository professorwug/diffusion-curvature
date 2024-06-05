# %% auto 0
__all__ = ['SadSpheres']

# %% ../nbs/library/datasets/saddle-sphere-ablations.ipynb 5
from .core import get_adaptive_graph
from .datasets import rejection_sample_from_saddle, sphere, plane
from .self_evaluating_datasets import SelfEvaluatingDataset, metric
from fastcore.all import *
import xarray as xr
import inspect
import pandas as pd

import sklearn
import scipy.stats
import numpy as np
import matplotlib.pyplot as plt


class SadSpheres(SelfEvaluatingDataset):
    def __init__(self,
                 dimension:list = [2], # Dimension of saddles and spheres. If a list is supplied, computed sadspheres for each
                 num_pointclouds = 100, # num pointclouds per dataset per dimension
                 num_points = 2000, # num points per pointclouds
                 noise_level = 0, # from 0 to 1. 1 is all noise.
                 include_planes = False, # if True, includes randomly sampled planes as a sanity check.
                ):
        store_attr()
        if isinstance(dimension, int):
            dimension = [dimension]

        
        datalist = []
        names = []
        
        for d in dimension:
            for i in range(num_pointclouds):
                X_saddle, ks_saddle = rejection_sample_from_saddle(self.num_points, d)
                datalist.append(
                    { 'X' : X_saddle, 'ks' : ks_saddle, 'd':d}
                )
                names.append(f'{d}-Saddle')
                
                X_sphere, ks_sphere = sphere(self.num_points, d)
                datalist.append(
                    { 'X' : X_sphere, 'ks' : ks_sphere[0], 'd':d}
                )
                names.append(f'{d}-Sphere')

                if self.include_planes:
                    X_plane = plane(self.num_points, d) 
                    X_plane = np.hstack([X_plane, np.zeros(self.num_points)[:,None]])
                    datalist.append(
                        { 'X' : X_plane, 'ks' : 0, 'd':d}
                    )
                    names.append(f'{d}-Plane')

        super().__init__(
            datalist, names, ['ks']
        )

                
    def get_item(self, idx):
        return self.DS[idx].obj['X']

    def get_truth(self, result_name, idx):
        truth = self.DS[idx].obj['ks']
        return truth

    def plot_by_dimension(self):
        names = self.names
        labels = self.labels['ks']
        # Extract unique dimensions
        dimensions = sorted(set(name[0] for name in names))
        
        # Number of methods
        methods = list(labels.keys())
        
        # Create a grid of plots
        fig, axs = plt.subplots(len(dimensions), len(methods), figsize=(5 * len(methods), 5 * len(dimensions)))
        if len(dimensions) == 1 or len(methods) == 1:
            axs = np.array(axs).reshape(len(dimensions), len(methods))
        
        # Define a color map based on unique dataset names without dimension prefix
        unique_names = sorted(set(name[2:] for name in names))  # Strip dimension prefix
        colors = plt.cm.get_cmap('viridis', len(unique_names))
        name_to_color = {name: colors(i) for i, name in enumerate(unique_names)}  # Correct mapping
        
        # Plotting
        for i, dim in enumerate(dimensions):
            for j, method in enumerate(methods):
                # Filter data for the current dimension
                data = [(labels[method][k], name[2:]) for k, name in enumerate(names) if name.startswith(dim)]  # Use name without prefix
        
                # Define bins for histogram
                all_values = [val for val, _ in data]
                bins = np.linspace(min(all_values), max(all_values), 51)  # 50 bins
        
                # Create histogram for each dataset
                ax = axs[i, j]
                for label in unique_names:
                    dataset_values = [val for val, name in data if name == label]  # Compare without dimension prefix
                    if dataset_values:  # Check if there are any values for this dataset
                        counts, _ = np.histogram(dataset_values, bins=bins)
                        bin_centers = 0.5 * (bins[:-1] + bins[1:])
                        ax.bar(bin_centers, counts, width=(bins[1] - bins[0]) * 0.9, color=name_to_color[label], label=label, alpha=0.75)
        
                ax.set_title(f'Dimension {dim} - {method}')
                ax.set_xlabel('Value Range')
                ax.set_ylabel('Count')
                if i == 0 and j == 0:  # Add legend only to the first subplot for clarity
                    ax.legend(title='Dataset Name')
        
        # Adjust layout
        plt.tight_layout()
        plt.show()
        

    

    @metric
    def pearson_r(self, a, b, result_name = None):
        return scipy.stats.pearsonr(a,b)[0]

    @metric
    def sign_score(self, 
                   a, # prediction
                   b, # target
                   result_name = None,
                  ):
        a = np.array(a)
        b = np.array(b)
        # measures classification accuracy of signs
        # First, get rid of zeros in ground truth curvatures; we don't want to classify the planes.
        nz = np.nonzero(b)[0]
        nonzero_preds = np.sign(a[nz])
        nonzero_targets = np.sign(b[nz])
        acc = np.sum((nonzero_preds == nonzero_targets).astype(int))/len(nz)
        return acc
        
            

