from ._cluster import strm_spectral
from ._utils import _check_iterable
from ._utils import calculate_rmse, weighted_kmeans, convert_clusterings_to_binary

from sklearn.cluster import KMeans
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

    # empty dictionary to hold the microclusters
    facilities = {}

    # generate microclusters
    for i in range(0, len(lowrankrange)):
        runs = strm_spectral(adata.X, num_clust, k=n_facility, n_parallel=n_parallel,
            streammode=True, svd_algorithm=svd_algorithm, 
            initial=initial, stream=stream, lowrankdim = lowrankrange[i])
        facilities.update(runs)

    # use microclusters to run different values of num_clust
    for K in num_clust:
        clusterings = _consolidate_microclusters(facilities, K)
        result = _combine_clustering_runs(clusterings, K)
        _write_results_to_anndata(result, adata, num_clust=K)


#from sklearn.cluster import KMeans
#import pandas as pd ???

def _write_results_to_anndata(result, adata, num_clust='result', prefix='sc3s_'):
    # mutating function
    # write clustering results to adata.obs, replacing previous results if they exist
    adata.obs = adata.obs.drop(prefix + str(num_clust), axis=1, errors='ignore')
    adata.obs.insert(len(adata.obs.columns), prefix + str(num_clust), result, allow_duplicates=False)

def _combine_clustering_runs(clusterings, num_clust):
    # combine clustering results using binary matrix method and k-means
    consensus_matrix = convert_clusterings_to_binary(clusterings)
    kmeans_macro = KMeans(n_clusters=num_clust, max_iter=10_000).fit(consensus_matrix)
    return pd.Categorical(kmeans_macro.labels_)

def _consolidate_microclusters(clusterings, num_clust):
    # take dict of clustering runs and consolidate with weighted k-means
    return {k: weighted_kmeans(run['cent'], run['asgn'], num_clust) for k, run in clusterings.items()}