from ._cluster import strm_spectral
from ._utils import _check_iterable
from ._utils import calculate_rmse
from ._utils import _write_results_to_anndata
from ._utils import _combine_clustering_runs_kmeans, _consolidate_microclusters
from ._utils import _combine_clustering_runs_hierarchical
import datetime

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


def consensus_clustering_legacy(
    adata, num_clust = [4],
    streaming = False, svd_algorithm = "sklearn",
    lowrankrange = range(10,20), n_parallel = 5,
    randomcellorder = True):

    assert _check_iterable(num_clust), "pls ensure num_clust is a list"

    time_start = datetime.datetime.now()
    print("start time:", time_start.strftime("%Y-%m-%d %H:%M:%S"))

    for K in num_clust:
        # empty dictionary to hold the clustering results
        clusterings = {}

        for i in range(0, len(lowrankrange)):
            # run spectral clustering, parameters modified to use the whole batch
            runs = strm_spectral(adata.X, k=K, n_parallel=n_parallel,
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

def get_num_range(num_clust, num_per_side=1, step=2, prefix = None):
    start = num_clust - num_per_side * step
    end = num_clust + num_per_side * step
    clust_range = list(filter(lambda x: x>1, range(start, end + 1, step)))
    
    if prefix is None:
        return clust_range
    else:
        return list(map(lambda x: prefix + str(x), clust_range))

