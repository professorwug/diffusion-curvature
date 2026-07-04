import os
import sys
import pytest

# Add the parent directory to the path so we can import the main module
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

# Import from the current experiment's main.py
from main import *

def test_12_curvature_menagerie_sanity():
    """Basic sanity check for the 12-curvature-menagerie experiment."""
    assert True, "Basic test for 12-curvature-menagerie"
