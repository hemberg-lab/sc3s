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

        # execute spectral clustering across different d values
        runs = _spectral_old(adata.X, k=K, d_range=lowrankrange, 
            n_runs=n_runs, svd=svd, return_centers=False)
    
        # extract the assignments as array form
        asgn_dict = {k: v['asgn'] for k, v in runs.items()}

        # perform the consensus clustering
        consensus_matrix = _make_contingency_consensus(asgn_dict)
        result = _cluster_contingency_consensus(consensus_matrix, K)

        # write results into adata.obs dataframe
        _write_results_to_anndata(result, adata, num_clust=K, prefix='sc3ori_')

    runtime = datetime.datetime.now() - time_start
    print("total runtime:", str(runtime))


def _make_contingency_consensus(asgn_dict):
    """
    Convert dictionary containing results for individual runs into a contingency matrix.
    """
    # check if assignments are correctly structured
    for x in asgn_dict.values():
        assert isinstance(x, np.ndarray) 

    X = np.array([x for x in asgn_dict.values()]).T   # row: cell, col: clustering run
    C = np.zeros((X.shape[0], X.shape[0]))
    for i in np.arange(0, X.shape[0]):
        C[i,] = np.sum(X[i,:] == X, axis=1) / X.shape[1]
    assert np.allclose(C, C.T), "contingency matrix not symmetrical"
    return C

def _cluster_contingency_consensus(consensus_matrix, num_clust):
    """
    Perform hierarchical clustering on a dictionary containing results for individual runs.
    """
    hclust = AgglomerativeClustering(n_clusters=num_clust, linkage='complete').fit(consensus_matrix)
    return pd.Categorical(hclust.labels_)