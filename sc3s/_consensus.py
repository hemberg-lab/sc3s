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

    # record start time and print to console
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

    # calculate number of microclusters
    n_facility = int(max(num_clust) * 8)
    assert n_facility <= stream

    # execute spectral clustering across different d values
    facilities = _spectral(adata.X, k = n_facility,
        d_range = lowrankrange,
        stream = stream, batch = batch,
        svd = svd, n_runs = n_runs, return_centers=True,
        randomcellorder = randomcellorder)

    for K in num_clust:
        print(f"running k = {K} ...")

        # cluster facilities into macroclusters
        # also extracts the assignments as array form
        asgn_dict = _cluster_facilities(facilities, K)

        # perform the consensus clustering
        consensus_matrix = _make_binary_consensus(asgn_dict)
        result = _cluster_binary_consensus(consensus_matrix, K)

        # write results into adata.obs dataframe
        _write_results_to_anndata(result, adata, num_clust=K)
    
    # record end time and print to console
    time_end = datetime.datetime.now()
    runtime = datetime.datetime.now() - time_start
    print(f"""
    END at {time_end.strftime("%Y-%m-%d %H:%M:%S")}

    total runtime: {str(runtime)}

    ======================================================================
    """)

def _cluster_facilities(facilities, num_clust):
    """Take dict of clustering runs and consolidate with weighted k-means."""
    # check if facilities dictionary is correctly structured
    assert isinstance(facilities, dict)
    for k, run in facilities.items():
        assert run.get('cent') is not None
        assert run.get('asgn') is not None

    # initialise empty dictionary for combined microclusters
    asgn_dict = {}

    for k, run in facilities.items():
        centroids = run['cent']
        assignments = run['asgn']

        (uniq_mclst, count_mclst) = np.unique(assignments, return_counts = True)

        # count the number of cells in each microcluster assignment
        weights = np.zeros(centroids.shape[0], dtype=int)
        weights[uniq_mclst] = count_mclst

        assert not np.any(np.isnan(centroids)), "NaNs in centroids"
        assert np.all(np.isfinite(centroids)), "Non-finite values in centroids"

        # add pseudoweight of 1, so microclusters with no cells assigned behave --- should drop these
        kmeans_weight = KMeans(n_clusters=num_clust, max_iter=1000).fit(centroids, sample_weight=weights+1) 
        macroclusters = kmeans_weight.labels_
        macrocentroids = kmeans_weight.cluster_centers_

        # update cell assignments in dictionary, remove the microcentroid centers
        asgn_dict[k] = macroclusters[assignments]
        
    return asgn_dict


def _make_binary_consensus(asgn_dict, datatype='float32'):
    """
    Converts clustering results into binary matrix for K-means.
    Requires that the number of data points are equal across clusterings.

    This updated version works even if the number of unique clusters are not the same.
    """
    # check if assignments are correctly structured
    for x in asgn_dict.values():
        assert isinstance(x, np.ndarray)

    def return_data_length_if_equal(asgn_dict):
        data_lengths = [len(x) for x in asgn_dict.values()]
        assert data_lengths.count(data_lengths[0]) == len(data_lengths), "data vectors different lengths"
        return data_lengths[0]

    def return_max_clustid(asgn_dict):
        maximum_values = [np.max(x) for x in asgn_dict.values()]
        return np.max(maximum_values) + 1

    # initialise binary matrix, after running some checks
    n_cells = return_data_length_if_equal(asgn_dict)
    n_clust = return_max_clustid(asgn_dict)
    B = np.zeros((n_cells, 0), dtype=datatype)
    results = list(asgn_dict.values())

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


def _cluster_binary_consensus(consensus_matrix, num_clust):
    kmeans_macro = KMeans(n_clusters=num_clust, max_iter=10_000).fit(consensus_matrix)
    return pd.Categorical(kmeans_macro.labels_)