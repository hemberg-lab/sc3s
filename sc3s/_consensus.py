from ._spectral import _spectral
from ._misc import _write_results_to_anndata, _parse_int_list
import datetime
import numpy as np
import pandas as pd
import anndata as ad
from sklearn.cluster import KMeans

def consensus(
    adata,
    num_clust = [4],
    n_facility = 100,
    lowrankrange = range(10, 20),
    stream = 1000,
    batch = 100,
    n_runs = 5,
    svd = "sklearn",
    randomcellorder = True,
    restart_chance = 0.05):

    # parse and check arguments
    num_clust = _parse_int_list(num_clust,
        error_msg = "num_clust must be integer > 1, or a non-empty list/range of such!")
    lowrankrange = _parse_int_list(lowrankrange,
        error_msg = "lowrankrange must be integer > 1, or a non-empty list/range of such!")

    assert isinstance(adata, ad.AnnData)
    assert isinstance(n_facility, int) and n_facility > max(num_clust)
    assert isinstance(stream, int)
    assert isinstance(batch, int)
    assert isinstance(n_runs, int)
    assert isinstance(randomcellorder, bool)
    assert 0 <= restart_chance <= 1

    # shrink stream and batch parameters if too large
    n_cells, n_genes = adata.X.shape
    if stream > n_cells:
        stream = n_cells
        print(f"Size of stream reduced to {stream}")

    if batch > stream:
        batch = stream
        print(f"Size of k-means batch reduced to {batch}")

    time_start = datetime.datetime.now()
    print(f"""
    ======================================================================
    SC3s CONSENSUS CLUSTERING

    size of dataset: {adata.shape[0]} obs, {adata.shape[1]} feature
    stream size: {stream}

    number of clusters: {num_clust}
    lowrankdim values: {lowrankrange}

    START at {time_start.strftime("%Y-%m-%d %H:%M:%S")}
    """)

    # empty dictionary to hold the microclusters
    facilities = {}

    # generate microclusters
    #for i in range(0, len(lowrankrange)):
    runs = _spectral(adata.X, k = n_facility,
                        d_range = lowrankrange,
                        stream = stream, batch = batch,
                        svd = svd, n_runs = n_runs,
                        randomcellorder = randomcellorder)
    facilities.update(runs)

    # use microclusters to run different values of num_clust
    for K in num_clust:
        print(f"running k = {K} ...")
        runs_dict = _consolidate_microclusters(facilities, K)
        result = _combine_clustering_runs_kmeans(runs_dict, K)
        _write_results_to_anndata(result, adata, num_clust=K)
    
    time_end = datetime.datetime.now()
    runtime = datetime.datetime.now() - time_start
    print(f"""
    END at {time_end.strftime("%Y-%m-%d %H:%M:%S")}

    total runtime: {str(runtime)}

    ======================================================================
    """)


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


def make_consensus_binary(runs_dict, datatype='float32'):
    """
    Converts clustering results into binary matrix for K-means.
    Requires that the number of data points are equal across clusterings.

    This updated version works even if the number of unique clusters are not the same.
    """
    def return_data_length_if_equal(runs_dict):
        data_lengths = [len(x) for x in runs_dict.values()]
        assert data_lengths.count(data_lengths[0]) == len(data_lengths), "data vectors different lengths"
        return data_lengths[0]

    def return_max_clustid(runs_dict):
        maximum_values = [np.max(x) for x in runs_dict.values()]
        return np.max(maximum_values) + 1

    # initialise binary matrix, after running some checks
    n_cells = return_data_length_if_equal(runs_dict)
    n_clust = return_max_clustid(runs_dict)
    B = np.zeros((n_cells, 0), dtype=datatype)
    results = list(runs_dict.values())

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


def _consolidate_microclusters(runs_dict, num_clust):
    # take dict of clustering runs and consolidate with weighted k-means
    return {k: weighted_kmeans(run['cent'], run['asgn'], num_clust) for k, run in runs_dict.items()}

def _combine_clustering_runs_kmeans(runs_dict, num_clust):
    # combine clustering results using binary matrix method and k-means
    consensus_matrix = make_consensus_binary(runs_dict)
    kmeans_macro = KMeans(n_clusters=num_clust, max_iter=10_000).fit(consensus_matrix)
    return pd.Categorical(kmeans_macro.labels_)