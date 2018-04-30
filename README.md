# DataStore

Use a DataStore object to manage a collection of data files, where each individual file fits in memory, but the entire collection does not necessarily fit.

## Example

DataStore objects assume that data file are organized in a tree-like structure. In this example, we will read the `.npy` files in the tree below:

![DataStore file tree](datastore_filetree.jpg)

Since these are Numpy `.npy` files, we can store the data in a `NumpyDataStore` object.

```python
from datastore import NumpyDataStore

ds = NumpyDataStore('MyData', include_subdirectories=True)
```

Here, we have passed the `NumpyDataStore` object the root directory for our file tree, and we have set `include_subdirectories=True`. As a result the DataStore will search all of the subdirectories
