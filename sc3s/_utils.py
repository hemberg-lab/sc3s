import numpy as np
import math
from scipy import linalg
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans

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

def weighted_kmeans(centroids, assignments, num_clust):
    """
    Weighted k means.
    """
    (uniq_mclst, count_mclst) = np.unique(assignments, return_counts = True)

    # count the number of cells in each microcluster assignment
    weights = np.zeros(centroids.shape[0], dtype=int)
    weights[uniq_mclst] = count_mclst

    assert not np.any(np.isnan(centroids)), "NaNs in centroids"
    assert np.all(np.isfinite(centroids)), "Non-finite values in centroids"

    kmeans_weight = KMeans(n_clusters=num_clust).fit(centroids, sample_weight=weights+1) # pseudoweight
    macroclusters = kmeans_weight.labels_
    macrocentroids = kmeans_weight.cluster_centers_

    return macroclusters[assignments]

def convert_clustering_to_binary(clustering, K):
    """
    Converts clustering results into binary matrix for K-means.
    
    Input: 
    * `clusterings`: `m * n` array of cluster assignments for `m` observations in `n` runs.
    * `k`: number of clusters

    Needs to be unit tested carefully. Doesn't throw error even if input is wrong orientation.
    Might need to incorporate checking for unique entries.
    """
    clusterings = np.transpose(clustering) # this should not exist
    n_parallel, n_cells = np.shape(clusterings)

    x = np.repeat(np.arange(0, n_cells), n_parallel)
    y = np.tile(np.arange(0, n_parallel), n_cells)
    z = clusterings.reshape(np.size(clusterings), order='F')

    B = np.zeros((n_cells, n_parallel, K), dtype=int)
    B[x,y,z] = 1
    B = B.reshape((n_cells, n_parallel*K))

    return B