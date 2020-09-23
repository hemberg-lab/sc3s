from ._spectral_old import _spectral_old
from ._misc import _write_results_to_anndata, _parse_int_list
import datetime
import anndata as ad
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering

def consensus_old(
    adata,
    num_clust = [4],
    lowrankrange = range(10,20),
    n_runs = 5,
    svd = "sklearn"):
    """
    Old version of SC3.
    """

    # parse and check arguments
    num_clust = _parse_int_list(num_clust,
        error_msg = "num_clust must be integer > 1, or a non-empty list/range of such!")
    lowrankrange = _parse_int_list(lowrankrange,
        error_msg = "lowrankrange must be integer > 1, or a non-empty list/range of such!")

    assert isinstance(adata, ad.AnnData)
    assert isinstance(n_runs, int)

    time_start = datetime.datetime.now()
    print("start time:", time_start.strftime("%Y-%m-%d %H:%M:%S"))

    for K in num_clust:
        print(f"running k = {K} ...")

        # empty dictionary to hold the clustering results
        runs_dict = _spectral_old(adata.X, k=K, d_range=lowrankrange, 
            n_runs=n_runs, svd=svd, return_centers=False)
    
        # perform the consensus clustering
        runs_dict = {k: v['asgn'] for k, v in runs_dict.items()}
        result = _cluster_contingency_consensus(runs_dict, K)
        _write_results_to_anndata(result, adata, num_clust=K, prefix='sc3ori_')

    runtime = datetime.datetime.now() - time_start
    print("total runtime:", str(runtime))

def _cluster_contingency_consensus(runs_dict, num_clust):
    """
    Perform hierarchical clustering on a dictionary containing results for individual runs.
    """
    # convert to contingency-based consensus, then does hierarchical clustering
    consensus_matrix = _make_contingency_consensus(runs_dict)
    hclust = AgglomerativeClustering(n_clusters=num_clust, linkage='complete').fit(consensus_matrix)
    return pd.Categorical(hclust.labels_)


def _make_contingency_consensus(runs_dict):
    """
    Convert dictionary containing results for individual runs into a contingency matrix.
    """
    X = np.array([x for x in runs_dict.values()]).T   # row: cell, col: clustering run
    print(X)
    print(X.shape)
    C = np.zeros((X.shape[0], X.shape[0]))
    for i in np.arange(0, X.shape[0]):
        C[i,] = np.sum(X[i,:] == X, axis=1) / X.shape[1]
    assert np.allclose(C, C.T), "contingency matrix not symmetrical"
    return C