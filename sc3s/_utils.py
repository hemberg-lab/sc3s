import numpy as np
import pandas as pd
import math
from scipy import linalg
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans, AgglomerativeClustering

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


def convert_clusterings_to_binary(clusterings, datatype='float32'):
    """
    Converts clustering results into binary matrix for K-means.
    Requires that the number of data points are equal across clusterings.
    
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

def convert_clusterings_to_contigency(clusterings):
    X = np.array([x for x in clusterings.values()]).T   # row: cell, col: clustering run
    C = np.zeros((X.shape[0], X.shape[0]))
    for i in np.arange(0, X.shape[0]):
        C[i,] = np.sum(X[i,:] == X, axis=1) / X.shape[1]
    assert np.allclose(C, C.T), "contingency matrix not symmetrical"
    return C

def _write_results_to_anndata(result, adata, num_clust='result', prefix='sc3s_'):
    # mutating function
    # write clustering results to adata.obs, replacing previous results if they exist
    adata.obs = adata.obs.drop(prefix + str(num_clust), axis=1, errors='ignore')
    adata.obs.insert(len(adata.obs.columns), prefix + str(num_clust), result, allow_duplicates=False)

def _check_iterable(obj):
    try:
        iter(obj)
    except Exception:
        return False
    else:
        return True


def _consolidate_microclusters(clusterings, num_clust):
    # take dict of clustering runs and consolidate with weighted k-means
    return {k: weighted_kmeans(run['cent'], run['asgn'], num_clust) for k, run in clusterings.items()}

def _combine_clustering_runs_kmeans(clusterings, num_clust):
    # combine clustering results using binary matrix method and k-means
    consensus_matrix = convert_clusterings_to_binary(clusterings)
    kmeans_macro = KMeans(n_clusters=num_clust, max_iter=10_000).fit(consensus_matrix)
    return pd.Categorical(kmeans_macro.labels_)

def _combine_clustering_runs_hierarchical(clusterings, num_clust):
    # convert to contingency-based consensus, then does hierarchical clustering
    consensus_matrix = convert_clusterings_to_contigency(clusterings)
    hclust = AgglomerativeClustering(n_clusters=num_clust, linkage='complete').fit(consensus_matrix)
    return pd.Categorical(hclust.labels_)