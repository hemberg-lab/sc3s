import numpy as np
import math
from scipy import linalg
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans as _skkmeams

def calculate_rmse(A, B):
    """
    Calculate root mean squared error between two matrices.
    """
    error = A - B
    return np.sum(error ** 2)


def svd_scipy(X, n_components):
    """
    Singular value decomposition using `scipy.linalg.svd`.
    Returned matrices are truncated to the value of `n_components`.
    """
    U, s, Vh = linalg.svd(X, full_matrices=False)
    U  = U[:, :n_components]
    s  = s[:n_components]
    Vh = Vh[:n_components, ]
    return U, s, Vh


def inv_svd(U, s, Vh):
    """
    Inverse of the singular value decomposition.
    """
    return np.dot(U, np.dot(np.diag(s), Vh))


def svd_sklearn(X, n_components, n_iter=5, random_state=None):
    """
    Truncated singular value decomposition using `scikitlearn`.
    """
    svd = TruncatedSVD(n_components, algorithm="randomized", n_iter=n_iter,
                       random_state=random_state)
    U = svd.fit_transform(X)
    s = svd.singular_values_
    Vh = svd.components_
    U = U / np.tile(s, (U.shape[0],1)) # by default, U is scaled by s
    return U, s, Vh
