# %% auto 0
__all__ = ['metric', 'Wrapper', 'SelfEvaluatingDataset']

# %% ../nbs/library/datasets/self-evaluating-datasets.ipynb 3
from fastcore.all import *
import inspect
import pandas as pd
from typing import List
import os

import sklearn
import scipy.stats
import numpy as np
import matplotlib.pyplot as plt
import joblib

def metric(func):
    setattr(func, 'tag', 'metric')
    return func

class Wrapper:
    def __init__(self, obj, **kwargs):
        self.obj = obj
        self.__dict__.update(kwargs)

class SelfEvaluatingDataset():
    def __init__(self,
                 datalist:List, # list of objects to be evaluated in the dataset. Usually includes multiple examples, e.g. a torus, sphere, saddle; multiple images, multiple validation datasets.
                 names:List, # names of the datasets in datalist.
                 result_names:List, # quantities to be computed (e.g. curvature, predictions). Usually just one per dataset.
                 save_directory:str = ".self-evaluating-datasets", 
                ):
        store_attr()

        self.DS = [ # list of datasets
            Wrapper(obj, results={rn:{} for rn in result_names}, name=name) for obj, name in zip(datalist, names)
        ]
        self.idx = -1
        for i in range(self.__len__()):
            # aggregate ground truth values
            for rn in self.result_names:
                self._store_truth(rn, i)

        # create the save directory if needed
        # Check if the directory exists
        if not os.path.exists(save_directory):
            # Create the directory
            os.makedirs(save_directory)
            self.imported_methods = []
        else:
            # try loading previous runs
            self.load_from_disk()

    
    def __iter__(self):
        return self

    def __len__(self):
        return len(self.DS)
    
    def preprocess(self, unprocessed_data_object):
        return unprocessed_data_object # override
    
    def get_item(self, idx):
        return self.DS[idx]

    def save_results(self):
        # saves DS to the save_directory using joblib dump
        child_name = type(self).__name__
        filename = os.path.join(self.save_directory, child_name + '.joblib_dump')
        joblib.dump(self.DS, filename)

    def load_from_disk(self):
        # loads saved DS from dict using joblib
        child_name = type(self).__name__
        filename = os.path.join(self.save_directory, child_name + '.joblib_dump')
        if os.path.exists(filename):
            self.DS = joblib.load(filename)
            r = self.DS[0].results
            self.imported_methods = list(r[list(r.keys())[0]].keys())
            print(f"Loaded evaluations on methods {self.imported_methods}")
        else:
            self.imported_methods = []
            print("No previously saved evaluations. Starting from 0.")
            
    def delete_saved_method(self, method_name):
        for i in range(len(self.DS)):
            for rn in self.result_names:
                self.DS[i].results[rn].pop(method_name)
        print("Deleted method from saved results")
        
    def __next__(self):
        self.idx += 1
        if self.idx >= self.__len__():
            self.save_results()
            raise StopIteration
        result = self.get_item(self.idx)
        return result

    def update(self,
               result,
               idx = None,
               result_name = 'default',
               method_name='computed',
               ):
        """
        Store the result of the curvature computation by passing the computed curvature of the center (first) point.
        """
        if idx is None: idx = self.idx
        if result_name == 'default': 
            if len(self.result_names) == 1:
                result_name = self.result_names[0]
        self.DS[idx].results[result_name][method_name] = result

    def get_truth(self, result_name, idx):
        """Compute the ground truth for each of your targets, and assign to a method. Usually this involves accessing some attribute of the input data and calling the update function"""
        truth = None
        return truth

    
    def _store_truth(self, result_name, idx):
        truth = self.get_truth(result_name, idx)
        self.update(
            truth, idx, method_name = "ground truth", result_name=result_name
        )


    def compute_metrics(self, labels, filter = None):
        metrics = self._get_metrics()
        metric_tables = {rn : {} for rn in self.result_names}
        for rn in self.result_names:
            for metric in metrics:
                metric_tables[rn][metric.__name__] = {}
                for method_name in self.method_names:
                    metric_tables[rn][metric.__name__][method_name] = self.compute(metric=metric, method_name=method_name, result_name=rn, labels = labels, filter = None)
            metric_tables[rn] = pd.DataFrame(metric_tables[rn])
        return metric_tables
            
    def compute(self, metric, result_name, method_name, labels, filter=None):
        # Overwrite this class with your logic. It implements the computation of a single metric for a single method
        return metric(labels[result_name][method_name], labels[result_name]['ground truth'], result_name = result_name)
    

    def _aggregate_labels(self):
        # returns a dictionary whose keys are method names, paired with a list of each of the results given by the metrics.
        # Just a more convenient data format for comparing method outputs.
        self.method_names = list(self.DS[0].results[self.result_names[0]].keys())
        self.labels = {}
        for rn in self.result_names:
            self.labels[rn] = {}
            for m in self.method_names:
                self.labels[rn][m] = [self.DS[i].results[rn][m] for i in range(self.__len__())]


    def plot(self, title = None):
        if title is None: title = f"In dimension {self.dimension}"
        # for each computed method on this dataset, we plot the histogram of saddles vs spheres
        self._aggregate_labels()
        # get the idxs for each type of dataset
        dataset_names = self.names
        unique_names = list(set(dataset_names))
        idxs_by_name = {n: [i for i, name in enumerate(dataset_names) if name == n] for n in unique_names}        
        for m in self.method_names: 
            if m != 'ks' and m != 'name':
                for dname in unique_names:
                    plt.hist(self.labels[m][idxs_by_name[dname]], bins=50, label = dname, edgecolor='none', linewidth=5)
                plt.legend()
                plt.xlabel(m)
                plt.title(title)
                plt.show()

    def table(self, filter=None):
        self._aggregate_labels()
        self.metric_tables = self.compute_metrics(filter=filter, labels = self.labels)
        for k in self.metric_tables.keys():
            print(k)
            print(self.metric_tables[k])
        return self.metric_tables

    def _get_metrics(self):
        tagged_functions = []
        methods = [method for method in dir(self) if callable(getattr(self, method))]
        for method_name in methods:
            member = getattr(self, method_name)
            if hasattr(member, 'tag') and getattr(member, 'tag') == 'metric':
                tagged_functions.append(member)
        return tagged_functions

    
