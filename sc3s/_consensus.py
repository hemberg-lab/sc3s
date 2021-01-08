import pandas
from sklearn.utils import check_random_state

from ._cluster import run_trials_miniBatchKMeans
from ._cluster import combine_facilities
from ._cluster import convert_dict_into_binary_matrix, cluster_consensus_matrix

from ._misc import _check_and_format_integer_list, _check_integer_single

import logging

def consensus(
    adata,
    n_clusters = [3, 5, 10],
    d_range = None,
    n_runs = 5,
    n_facility = None,
    multiplier_facility = None,
    batch_size = 100,
    random_state = None):

    logging.basicConfig(level=logging.INFO)

    # check that AnnData object already has PCA coordinates
    try:
        X_pca = adata.obsm['X_pca']
    except:
        raise Exception("AnnData object does not have PCA coordinates. Please run that first.")

    # get the number of cells from row dimension
    n_cells = X_pca.shape[0]

    # check and formats n_clusters into in a list format, using default values if unprovided
    n_clusters = _check_and_format_integer_list(
        n_clusters, 
        min_val = 2, 
        max_val = n_cells,
        var_name = 'n_clusters'
    )

    # check d_range
    d_range = _check_and_format_integer_list(
        d_range,
        min_val = 2,
        max_val = X_pca.shape[1],
        var_name = 'd_range'
    )

    # check n_runs
    n_runs = _check_integer_single(n_runs, min_val=1, var_name="n_runs")

    # check batch_size
    if isinstance(batch_size, int):
        assert batch_size <= n_cells, "batch size for k-means must be smaller than number of cells."
    else:
        raise Exception("batch size must be positive integer value (default: 100)")


    # check random state
    random_state = check_random_state(random_state)
 
    # determine number of facilities to use
    if n_facility is not None:
        logging.info(f"n_facility set to {n_facility}, ignoring multiplier_facility parameter...")
    else:
        if multiplier_facility is None:
            logging.info(f"multiplier_facility not set, using value of 5...")
            multiplier_facility = 3
        # calculate the number of facilities, if not provided
        if max(n_clusters) * multiplier_facility <= n_cells:
            n_facility = max(n_clusters) * multiplier_facility
            logging.info(f"number of facilities calculated as {n_facility}")
    
    # if the number of facilities is too high, raise an error
    # edge case: the sample has very few cells and user does not specify n_facility
    n_facility = _check_integer_single(
        n_facility, 
        min_val = 2, 
        max_val = n_cells,
        var_name = 'n_facility'
    )

    # run over many different trials
    trials_dict = run_trials_miniBatchKMeans(
        data = X_pca,
        n_clusters = n_facility,
        d_range = d_range,
        n_runs = n_runs,
        batch_size = batch_size,
        random_state = random_state
    )

    # write the individual trials into AnnData object
    adata.uns['sc3s_trials'] = trials_dict

    # run different number of clusters
    for K in n_clusters:
        consensus_dict = combine_facilities(
            dict_object = trials_dict, 
            K = K, 
            n_facility = n_facility,
            batch_size = batch_size, 
            random_state = random_state
        )

        B = convert_dict_into_binary_matrix(
            dict_object = consensus_dict,
            true_n_clusters = K,
            true_n_cells = n_cells
        )

        cell_labels = cluster_consensus_matrix(
            consensus_matrix = B,
            n_clusters = K,
            batch_size = batch_size,
            random_state = random_state
        )

        # write results into AnnData object
        adata.obs[f'sc3s_{K}'] = cell_labels
        adata.obs[f'sc3s_{K}'] = adata.obs[f'sc3s_{K}'].astype('category')

    return adata