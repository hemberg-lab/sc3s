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
    batch_size = None,
    random_state = None):
    """\
    Run consensus clustering to cluster cells in an AnnData object.
    
    This function requires the user to perform dimensionality
    reduction using PCA (`scanpy.tl.pca`) first.

    Parameters
    ----------
    adata
        Annotated data matrix.
    n_clusters
        Number of clusters. Default: [3,5,10]
    d_range
        Number of PCs. Default is 25, or the number of PCs in the 
        AnnData object, whichever is lower. Can accept a list
        (e.g. `[15, 20, 25`]).
    n_runs
        Number of realisations to perform for the consensus.
        Default is 5, recommended > 1.
    n_facility
        Number of microclusters. Overridden if `multiplier_facility`
        is provided with a value.
    multiplier_facility
        Multiplier for microclusters. Number of microclusters
        is calculated as this parameter multiplied by the max of
        `n_clusters`. Default is 3.
    batch_size
        Batch size for k-means. Default is 100.
    random_state
        Random state of the algorithm.

    Returns
    -------
    AnnData object with cluster labels written in the `.obs` dataframe.

    adata.obs['sc3s_{k}']
        Labels for `n_clusters = k`.
    """

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
    if d_range is None:
        d_range = [min(25, X_pca.shape[1])]
    d_range = _check_and_format_integer_list(
        d_range,
        min_val = 2,
        max_val = X_pca.shape[1],
        var_name = 'd_range'
    )

    # check n_runs
    n_runs = _check_integer_single(n_runs, min_val=1, var_name="n_runs")

    # check batch_size
    if batch_size is None:
        batch_size=min(100, adata.shape[0])
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
            logging.info(f"multiplier_facility not set, using default value of 3...")
            multiplier_facility = 3
        
        # use multiplier value to calculate
        n_facility = max(n_clusters) * multiplier_facility

        if n_facility >= n_cells:
            n_facility = n_cells - 1

        logging.info(f"number of facilities calculated as {n_facility}")    
    
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