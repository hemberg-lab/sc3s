from ._cluster import strm_spectral
from ._utils import _check_iterable
from ._utils import calculate_rmse
from ._utils import _write_results_to_anndata
from ._utils import _combine_clustering_runs_kmeans, _consolidate_microclusters
from ._utils import _combine_clustering_runs_hierarchical
import datetime
import numpy as np
import pandas as pd

def consensus_clustering(
    adata, num_clust = [4], n_facility = 100,
    streaming = True, svd_algorithm = "sklearn",
    initial = 100, stream = 50,
    lowrankrange = range(10,20), n_parallel = 5,
    initialmin = 10**3, streammin = 10,
    initialmax = 10**5, streammax = 100,
    randomcellorder = True):

    assert _check_iterable(num_clust), "pls ensure num_clust is a list"

    time_start = datetime.datetime.now()
    print("start time:", time_start.strftime("%Y-%m-%d %H:%M:%S"))

    # empty dictionary to hold the microclusters
    facilities = {}

    # generate microclusters
    for i in range(0, len(lowrankrange)):
        runs = strm_spectral(adata.X, k=n_facility, n_parallel=n_parallel,
            streammode=True, svd_algorithm=svd_algorithm, 
            initial=initial, stream=stream, lowrankdim = lowrankrange[i])
        facilities.update(runs)

    # use microclusters to run different values of num_clust
    for K in num_clust:
        clusterings = _consolidate_microclusters(facilities, K)
        result = _combine_clustering_runs_kmeans(clusterings, K)
        _write_results_to_anndata(result, adata, num_clust=K)
    
    runtime = datetime.datetime.now() - time_start
    print("total runtime:", str(runtime))


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


def _consolidate_microclusters(clusterings, num_clust):
    # take dict of clustering runs and consolidate with weighted k-means
    return {k: weighted_kmeans(run['cent'], run['asgn'], num_clust) for k, run in clusterings.items()}

def _combine_clustering_runs_kmeans(clusterings, num_clust):
    # combine clustering results using binary matrix method and k-means
    consensus_matrix = convert_clusterings_to_binary(clusterings)
    kmeans_macro = KMeans(n_clusters=num_clust, max_iter=10_000).fit(consensus_matrix)
    return pd.Categorical(kmeans_macro.labels_)