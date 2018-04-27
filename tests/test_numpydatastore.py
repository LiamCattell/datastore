import sys
sys.path.append('../../datastore')

import numpy as np
from datastore import NumpyDataStore

root = 'X:/liam/experiments/2018-04_lot_theory/data/gaussians/lot'
include_subdirectories = True

ds = NumpyDataStore(root, include_subdirectories)

print(ds.n_classes, ds.labels)

X, y = ds.load(['class1', 'class3'], [0,2])
print(len(X), X[0].shape, y)

# l,i = ds._check_load_inputs(['class1','lot'], [0,1])
# print(l, i)
