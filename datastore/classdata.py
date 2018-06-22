import numpy as np
import h5py
from os import path
from scipy.io import loadmat
from imageio import imread


class ClassData(object):
    """
    Base ClassData class.

    This object stores the filepaths of a dataset and can load in data when
    required.

    Parameters
    ----------
    filepaths : list of str
        Filepaths to store

    Attributes
    ----------
    files : list of str
        Data filepaths
    n_files : int
        Number of files stored
    """
    def __init__(self, filepaths):
        self.files = filepaths
        self.n_files = len(filepaths)
        return


    def __getitem__(self, key):
        """
        Override [] operator so we can load data at a given index.
        """
        return self.load(ind=key)


    def _is_iterable(self, x):
        """
        Check if x is a list, tuple, or numpy array.
        """
        return isinstance(x,list) or isinstance(x,tuple) or \
               isinstance(x,np.ndarray)


    def _check_ind(self, ind):
        """
        Check that ind are valid indices that can be loaded.

        Parameters
        ----------
        ind : int or list/tuple/array of int
            Indices of data

        Returns
        -------
        ind_out : list of int
            Valid indices
        """
        # Convert ind to list of indices
        if ind is None:
            ind_out = range(self.n_files)
        elif not self._is_iterable(ind):
            ind_out = [ind]
        else:
            ind_out = ind

        # Check that indices are within array range
        if max(ind_out) >= self.n_files or min(ind_out) < 0:
            raise ValueError("Index {} out of bounds.".format(max(ind_out)))

        return ind_out


    def _load_func(self, filepath):
        """
        The function used to load the data stored in self.files.
        """
        raise NotImplementedError("ClassData is a base class, so no "
                                  "_load_func() is defined.")
        return


    def load(self, ind=None):
        """
        Load the data.

        Parameters
        ----------
        ind : int or list/tuple/array of int or None (default=None)
            Indices of the data to be loaded. If None, all of the data are
            loaded

        Returns
        -------
        data : data object or list of data object
            Loaded data
        """
        ind = self._check_ind(ind)

        data = []
        for i in ind:
            if path.isfile(self.files[i]):
                data.append(self._load_func(self.files[i]))

        if len(data) == 1:
            return data[0]
        else:
            return data


class NumpyClassData(ClassData):
    """
    ClassData class for '.npy' files.

    This object stores the filepaths of a '.npy' dataset and can load in files
    when required.

    Parameters
    ----------
    filepaths : list of str
        Filepaths to store

    Attributes
    ----------
    files : list of str
        Data filepaths
    n_files : int
        Number of files stored
    """
    def _load_func(self, filepath):
        """
        Function used to load the data.

        Parameters
        ----------
        filepath : str
            Filepath of '.npy' file

        Returns
        -------
        data : numpy.ndarray
            Loaded data file
        """
        return np.load(filepath)


class MatlabClassData(ClassData):
    """
    ClassData class for '.mat' Matlab files.

    This object stores the filepaths of a '.mat' dataset and can load in files
    when required.

    Parameters
    ----------
    filepaths : list of str
        Filepaths to store

    Attributes
    ----------
    files : list of str
        Data filepaths
    n_files : int
        Number of files stored
    """
    def _load_func(self, filepath):
        """
        Function used to load the data.

        Parameters
        ----------
        filepath : str
            Filepath of '.mat' file

        Returns
        -------
        data : numpy.ndarray
            Loaded data file
        """
        return loadmat(filepath)


class HDF5ClassData(ClassData):
    """
    ClassData class for HDF5 (and Matlab v7.3) files

    This object stores the filepaths of a HDF5 dataset and can load in files
    when required.

    Parameters
    ----------
    filepaths : list of str
        Filepaths to store

    Attributes
    ----------
    files : list of str
        Data filepaths
    n_files : int
        Number of files stored
    """
    def _load_func(self, filepath):
        """
        Function used to load the data.

        Parameters
        ----------
        filepath : str
            Filepath of HDF5 (or Matlab v7.3) file

        Returns
        -------
        data : h5py.File object
            Loaded data file
        """
        return h5py.File(filepath, 'r')


class ImageClassData(ClassData):
    """
    ClassData class for image files.

    This object stores the filepaths of an image dataset and can load in files
    when required.

    Parameters
    ----------
    filepaths : list of str
        Filepaths to store

    Attributes
    ----------
    files : list of str
        Data filepaths
    n_files : int
        Number of files stored
    """
    def _load_func(self, filepath):
        """
        Function used to load the data.

        Parameters
        ----------
        filepath : str
            Filepath of image file

        Returns
        -------
        data : ndarray
            Loaded data file
        """
        return imread(filepath, as_gray=True)
