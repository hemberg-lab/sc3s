import warnings
import pandas
from sklearn.utils import check_random_state
from . import cluster_binary_matrix, combine_facilities_in_trials, convert_trials_into_binary_matrix, run_trials_miniBatchKMeans
from . import _format_integer_list_to_iterable 

def consensus_clustering(adata,
                         n_clusters = [3, 5, 10],
                         n_facility = None,
                         d_range = None,
                         n_runs = 5,
                         batch_size = 100,
                         random_state = None):

    n_cells = adata.shape[0]

    # check that AnnData object already has PCA coordinates
    try:
        X_pca = adata.obsm['X_pca']
    except:
        raise Exception("AnnData object does not have PCA coordinates. Please run that first.")

    # check and formats n_clusters into in a list format, using default values if unprovided
    if n_clusters is None:
        raise Exception("Please specify value for n_clusters.")
    else:
        n_clusters = _format_integer_list_to_iterable(n_clusters, var_name = "n_clusters")
        for i in n_clusters:
            if i >= n_cells:
                raise Exception("Values for n_clusters must be fewer than number of cells.")

    # check d_range
    # must be less than size of PCA object

    # check n_runs
    assert isinstance(n_runs, int) and n_runs > 1, "n_runs must be positive integer value."

    # check batch_size
    if isinstance(batch_size, int):
        assert batch_size <= n_cells, "batch size for k-means must be smaller than number of cells."
    else:
        raise Exception("batch size must be positive integer value (default: 100)")


    # check random state


    # calculate the number of facilities
    if n_facility is None:
        if max(n_clusters) * 10 <= n_cells: 
            n_facility = max(n_clusters) * 10
        else:
            warnings.warn(f"There isn't many cells, n_facility set to maximum n_clusters value: {max(n_clusters)}")
            n_facility = max(n_clusters)


    # run over many different trials
    trials_dict = run_trials_miniBatchKMeans(data = X_pca,
                                             n_clusters = n_clusters,
                                             d_range = d_range,
                                             n_runs = n_runs,
                                             batch_size = batch_size,
                                             random_state = random_state)

    # run different number of clusters
    for K in n_clusters:
        consensus_dict = combine_facilities(dict_object = trials_dict, 
                                            K = K, 
                                            batch_size = batch_size, 
                                            random_state = random_state)

        B = convert_trials_into_binary_matrix(dict_object = consensus_dict,
                                              true_n_clusters = K,
                                              true_n_cells = n_cells)

        cell_labels = cluster_binary_matrix(binary_consensus_matrix = B,
                                            n_clusters = K,
                                            batch_size = batch_size,
                                            random_state = random_state)
    
        # write results into AnnData object
        adata.obs[f'sc3s_{K}'] = cell_labels
        adata.obs[f'sc3s_{K}'] = adata.obs[f'sc3s_{K}'].astype('category')
