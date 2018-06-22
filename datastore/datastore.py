import os
import warnings
import numpy as np

from .classdata import *


class DataStore(object):
    """
    Base DataStore class.

    Use a DataStore object to manage a collection of data files, where each
    individual file fits in memory, but the entire collection does not
    necessarily fit.

    Parameters
    ----------
    root : str
        Root directory from which to load data
    extension : str
        File extension for the desired files. e.g '.npy'
    include_subdirectories : bool (default=True)
        If False, files are only loaded from the root directory. If True, files
        are loaded from the root directory and all subdirectories of root.
    cdtype : ClassData object or None:
        Defines the type of ClassData object used to store/load the files.

    Attributes
    ----------
    labels : dict
        Dictionary of class labels and corresponding data. Keys are label names
        and values are ClassData objects.
    n_classes : int
        Number of classes.
    n_files : list of int
        Number of files in each class.
    root : str
        Path of the root directory.
    extension : str or list of str
        File extension(s) of the stored data.
    """
    def __init__(self, root, extension, include_subdirectories=True,
                 cdtype=None):

        # Convert relative root path to absolute path
        root = os.path.abspath(root)

        # Convert extension to list
        if isinstance(extension, str):
            extension = [extension]

        # Check extension is something like '.jpg', not 'jpg'
        extension = ['.{}'.format(x) if x[0] != '.' else x for x in extension]

        # Make extensions lowercase
        extension = [x.lower() for x in extension]

        if cdtype is None:
            self.cdtype = ClassData
        else:
            self.cdtype = cdtype

        # Initialise class label names
        self.labels = {}

        # Get filetree
        filetree = os.walk(root)

        # Get files in root
        dirpath, _, _ = next(filetree)
        self._add_files(dirpath, extension)

        # Add files in subdirectories
        if include_subdirectories:
            # Loop over all subdirectories in root
            for subdir in sorted(filetree):
                self._add_files(subdir[0], extension)

        # Warn the user if no files were found
        if not self.labels:
            warnings.warn("No '{}' files found in {}.".format(extension, root))

        # Set up other useful attributes
        self.root = root
        self.extension = extension
        self.n_classes = len(self.labels)
        self.n_files = []
        for v in self.labels.values():
            self.n_files.append(v.n_files)

        return


    def __getitem__(self, key):
        """
        Override the [] operator so the user can return the ClassData object
        for a given data class.
        """
        if not key in self.labels:
            raise ValueError("Label '{}' does not exist.".format(l))

        return self.labels[key]


    def _add_files(self, dirpath, extension):
        """
        Find files in a given directory, and create a ClassData object for that
        data class (as long as files exist).

        Parameters
        ----------
        dirpath : str
            Directory in which to search for files
        extension : str
            File extension for the desired files. e.g '.npy'
        """
        # Loop over files in the directory
        filepaths = []
        for f in sorted(os.listdir(dirpath)):
            # If this file matches the filetype, add it to the list
            if os.path.splitext(f)[1].lower() in extension:
                filepaths.append(os.path.join(dirpath,f))

        # If filepaths exist, update the dict of labels
        if filepaths:
            self.labels[os.path.basename(dirpath)] = self.cdtype(filepaths)

        return


    def _is_iterable(self, x):
        """
        Check if x is a list, tuple, or numpy array
        """
        return isinstance(x,list) or isinstance(x,tuple) or \
               isinstance(x,np.ndarray)


    def _check_label(self, label):
        """
        Compare label to self.labels and check that they exist.

        Parameters
        ----------
        label : str or list/tuple/array of str
            Label(s) to check

        Returns
        -------
        lab_out : list of str
            List of valid label(s)
        """
        # Create a list of labels based on the input label variable
        if label is None:
            lab_out = list(self.labels.keys())
        elif not self._is_iterable(label):
            lab_out = [label]
        else:
            lab_out = label

        # Check that elements in lab_out are valid labels
        for l in lab_out:
            if not l in self.labels:
                raise ValueError("Label '{}' does not exist.".format(l))

        return lab_out


    def load(self, label=None, ind=None):
        """
        Load data in the DataStore.

        Parameters
        ----------
        label : str or list/tuple/array of str or None (default=None)
            Class label(s) to load. If None, all labels are loaded
        ind : int or list/tuple/array of int or None (default=None)
            Indices of the data to be loaded. If None, all data are loaded for
            the given class(es)

        Returns
        -------
        data : data object or list of data objects
            Loaded data
        labels : str or list of str
            Class labels corresponding to data
        """
        # Check that the input label is valid
        label = self._check_label(label)

        # Initialise output lists
        data = []
        labels = []

        # Loop over each desired label and load the data
        for l in label:
            tmp = self.labels[l].load(ind)
            if isinstance(tmp, list):
                data += tmp
                labels += [l] * len(tmp)
            else:
                data.append(tmp)
                labels.append(l)

        # Should we return a list or a single data point?
        if len(data) == 1:
            return data[0], labels[0]
        else:
            return data, labels


class NumpyDataStore(DataStore):
    """
    DataStore object for '.npy' data.

    Use a NumpyDataStore object to manage a collection of '.npy' files, where
    each individual file fits in memory, but the entire collection does not
    necessarily fit.

    Parameters
    ----------
    root : str
        Root directory from which to load data
    include_subdirectories : bool (default=True)
        If False, files are only loaded from the root directory. If True, files
        are loaded from the root directory and all subdirectories of root.
    """
    def __init__(self, root, include_subdirectories=True):
        super(NumpyDataStore, self).__init__(root, '.npy',
                                             include_subdirectories,
                                             cdtype=NumpyClassData)
        return


class MatlabDataStore(DataStore):
    """
    DataStore object for '.mat' Matlab data.

    Use a NumpyDataStore object to manage a collection of '.mat' files, where
    each individual file fits in memory, but the entire collection does not
    necessarily fit.

    Parameters
    ----------
    root : str
        Root directory from which to load data
    include_subdirectories : bool (default=True)
        If False, files are only loaded from the root directory. If True, files
        are loaded from the root directory and all subdirectories of root.
    """
    def __init__(self, root, include_subdirectories=True):
        super(MatlabDataStore, self).__init__(root, '.mat',
                                              include_subdirectories,
                                              cdtype=MatlabClassData)
        return


class HDF5DataStore(DataStore):
    """
    DataStore object for HDF5 (and Matlab v7.3) data.

    Use a NumpyDataStore object to manage a collection of HDF5 files, where
    each individual file fits in memory, but the entire collection does not
    necessarily fit.

    Parameters
    ----------
    root : str
        Root directory from which to load data
    include_subdirectories : bool (default=True)
        If False, files are only loaded from the root directory. If True, files
        are loaded from the root directory and all subdirectories of root.
    """
    def __init__(self, root, include_subdirectories=True):
        super(HDF5DataStore, self).__init__(root, '.mat',
                                            include_subdirectories,
                                            cdtype=HDF5ClassData)
        return


class ImageDataStore(DataStore):
    """
    DataStore object for image data.

    Use a NumpyDataStore object to manage a collection of image files, where
    each individual file fits in memory, but the entire collection does not
    necessarily fit.

    Parameters
    ----------
    root : str
        Root directory from which to load data
    include_subdirectories : bool (default=True)
        If False, files are only loaded from the root directory. If True, files
        are loaded from the root directory and all subdirectories of root.
    """
    def __init__(self, root, include_subdirectories=True):
        extensions = ['.jpg', '.png', '.tif', '.bmp']
        super(ImageDataStore, self).__init__(root, extensions,
                                             include_subdirectories,
                                             cdtype=ImageClassData)
        return
