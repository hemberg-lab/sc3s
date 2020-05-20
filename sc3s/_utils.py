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

    kmeans_weight = KMeans(n_clusters=num_clust, max_iter=1000).fit(centroids, sample_weight=weights+1) # pseudoweight
    macroclusters = kmeans_weight.labels_
    macrocentroids = kmeans_weight.cluster_centers_

    return macroclusters[assignments]

def _convert_clustering_to_binary_deprecated(clustering, K):
    """
    OLD VERSION. TO BE DEPRECATED.
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

def convert_clusterings_to_binary(clusterings, datatype='float32'):
    """
    Converts clustering results into binary matrix for K-means.
    Requires that the number of data points are equal across clusterings, and 
    that each point only has one assignment.
    
    Input: 
    * `clusterings`: a dictionary

    This updated version works even if the number of unique clusters are not the same.
    """
    def return_data_length_if_equal(clusterings):
        data_lengths = [len(x) for x in clusterings.values()]
        assert data_lengths.count(data_lengths[0]) == len(data_lengths), "data vectors different lengths"
        return data_lengths[0]

    def return_max_clustid(clusterings):
        maximum_values = [np.max(x) for x in clusterings.values()]
        return np.max(maximum_values) + 1

    # initialise binary matrix, after running some checks
    n_cells = return_data_length_if_equal(clusterings)
    n_clust = return_max_clustid(clusterings)
    B = np.zeros((n_cells, 0), dtype=datatype)
    results = list(clusterings.values())

    # fill in the binary matrix
    for i in range(0, len(results)):
        b = np.zeros((n_cells, n_clust), dtype=datatype)
        b[range(0, n_cells), results[i]] = 1
        assert np.all(np.sum(b, axis=1) == 1), "some data points have multiple cluster assignments"
        B = np.append(B, b, axis=1)

    assert B.shape[1] == (len(results) * n_clust), "binary matrix has wrong number of columns"

    # remove clusters with no cells assigned
    # this can happen if some initialised centroids do not get assigned anything
    print("original binary matrix:", B.shape)
    B = B[:,np.sum(B, axis=0)!=0]
    print("removed clusters with no assignments, now:", B.shape, "\n")

    return B