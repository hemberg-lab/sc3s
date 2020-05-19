from ._cluster import strm_spectral
from ._utils import calculate_rmse, weighted_kmeans, convert_clustering_to_binary

from sklearn.cluster import KMeans
import numpy as np
import pandas as pd

def consensus_clustering(
    adata, num_clust = 4, 
    streaming = True, svd_algorithm = "sklearn",
    initial = 0.2, stream = 0.02,
    lowrankrange = range(10,20), n_parallel = 5,
    initialmin = 10**3, streammin = 10,
    initialmax = 10**5, streammax = 100,
    randomcellorder = True):

    n_cells, n_genes = adata.X.shape

    # empty dictionary to hold clustering results
    #clusterings = np.empty((n_cells, len(lowrankrange), n_parallel), dtype=int)
    clusterings = {}

    #np.random.seed(322)
    for i in range(0, len(lowrankrange)):
        runs = strm_spectral(adata.X, num_clust, k=100, 
            streammode=True, svd_algorithm=svd_algorithm, 
            initial = 100, stream = 20, lowrankdim = lowrankrange[i])
        clusterings.update(runs)

        #clusterings[:, i, j] = weighted_kmeans(microcentroids, assignments)

    
    """
    # convert to binary matrix
    clusterings = clusterings.reshape((n_cells, len(lowrankrange)*n_parallel))
    consensus_matrix = convert_clustering_to_binary(clusterings, num_clust)

    # consolidate microclusters
    kmeans_macro = KMeans(n_clusters=num_clust).fit(consensus_matrix)

    # write clustering results to adata.obs, replacing previous results if they exist
    adata.obs = adata.obs.drop("sc3s", axis=1, errors='ignore')
    adata.obs.insert(len(adata.obs.columns), "sc3s", 
        pd.Categorical(kmeans_macro.labels_), allow_duplicates=True)
    """

    return clusterings #adata

