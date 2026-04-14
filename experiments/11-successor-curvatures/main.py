"""
Main module for the 11-successor-curvatures experiment.

This module contains the core functionality for the experiment.
"""
from snakemake.script import snakemake # direct access to Snakefile variables
data_input = snakemake.input.data

print("Running 11-successor-curvatures …")
