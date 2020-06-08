from ._cluster import strm_spectral
from ._utils import _check_iterable
from ._utils import calculate_rmse, weighted_kmeans, convert_clusterings_to_binary

from sklearn.cluster import KMeans
import numpy as np
import pandas as pd

def consensus_clustering(
    adata, num_clust = [4], 
    streaming = True, svd_algorithm = "sklearn",
    initial = 100, stream = 50,
    lowrankrange = range(10,20), n_parallel = 5,
    initialmin = 10**3, streammin = 10,
    initialmax = 10**5, streammax = 100,
    randomcellorder = True):

    assert _check_iterable(num_clust), "pls ensure num_clust is a list"

    # empty dictionary to hold clustering results
    clusterings = {}

    # generate microclusters
    for i in range(0, len(lowrankrange)):
        runs = strm_spectral(adata.X, num_clust, k=100, n_parallel=n_parallel,
            streammode=True, svd_algorithm=svd_algorithm, 
            initial=initial, stream=stream, lowrankdim = lowrankrange[i])
        clusterings.update(runs)

    for K in num_clust:
        # consolidate microclusters
        clusterings_macro = {k: weighted_kmeans(run['cent'], run['asgn'], K) for k, run in runs.items()}

        # convert to binary matrix
        consensus_matrix = convert_clusterings_to_binary(clusterings_macro)
        
        # consolidate macroclustering runs
        kmeans_macro = KMeans(n_clusters=K, max_iter=5000).fit(consensus_matrix)

        # write clustering results to adata.obs, replacing previous results if they exist
        adata.obs = adata.obs.drop("sc3s", axis=1, errors='ignore')
        adata.obs.insert(len(adata.obs.columns), "sc3s_" + str(K), 
            pd.Categorical(kmeans_macro.labels_), allow_duplicates=True)