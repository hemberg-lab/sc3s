from ._spectral import _spectral
from ._misc import _write_results_to_anndata
import datetime
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering

def consensus_old(
    adata, num_clust = [4],
    streaming = False, svd_algorithm = "sklearn",
    lowrankrange = range(10,20), n_parallel = 5,
    randomcellorder = True):
    """
    Old version of SC3.
    """

    num_clust = _parse_int_list(num_clust, error_msg = "num_clust must be integer > 1, or a non-empty list/range of such!")
    lowrankrange = _parse_int_list(lowrankrange, error_msg = "lowrankrange must be integer > 1, or a non-empty list/range of such!")

    time_start = datetime.datetime.now()
    print("start time:", time_start.strftime("%Y-%m-%d %H:%M:%S"))

    for K in num_clust:
        # empty dictionary to hold the clustering results
        clusterings = {}

        for i in range(0, len(lowrankrange)):
            # run spectral clustering, parameters modified to use the whole batch
            runs = _spectral(adata.X, k=K, n_parallel=n_parallel,
                streammode=False, svd_algorithm=svd_algorithm, initial=adata.X.shape[0],
                initialmin=adata.X.shape[0], initialmax=adata.X.shape[0], # just so it behaves
                lowrankdim = lowrankrange[i])
            clusterings.update(runs)

        # perform the consensus clusterings
        clusterings = {k: v['asgn'] for k, v in clusterings.items()}
        result = _combine_clustering_runs_hierarchical(clusterings, K)
        _write_results_to_anndata(result, adata, num_clust=K, prefix='sc3ori_')

    runtime = datetime.datetime.now() - time_start
    print("total runtime:", str(runtime))

def _combine_clustering_runs_hierarchical(clusterings, num_clust):
    """
    Perform hierarchical clustering on a dictionary containing results for individual runs.
    """
    # convert to contingency-based consensus, then does hierarchical clustering
    consensus_matrix = convert_clusterings_to_contigency(clusterings)
    hclust = AgglomerativeClustering(n_clusters=num_clust, linkage='complete').fit(consensus_matrix)
    return pd.Categorical(hclust.labels_)


def convert_clusterings_to_contigency(clusterings):
    """
    Convert dictionary containing results for individual runs into a contingency matrix.
    """
    X = np.array([x for x in clusterings.values()]).T   # row: cell, col: clustering run
    C = np.zeros((X.shape[0], X.shape[0]))
    for i in np.arange(0, X.shape[0]):
        C[i,] = np.sum(X[i,:] == X, axis=1) / X.shape[1]
    assert np.allclose(C, C.T), "contingency matrix not symmetrical"
    return C