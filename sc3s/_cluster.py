from ._misc import rk, rv
from ._misc import _check_and_format_integer_list

def run_trials_miniBatchKMeans(data, n_clusters, d_range, n_runs, batch_size, random_state):
    """
    Generate dictionary object containing MiniBatchKMeans clustering runs.
    """

    import itertools
    from sklearn.cluster import MiniBatchKMeans

    # n_clusters must be single integer

    d_range = _check_and_format_integer_list(
        d_range,
        min_val = 2,
        max_val = data.shape[1],
        var_name = 'd_range'
    )

    assert isinstance(n_runs, int) and n_runs > 1, "n_runs must be positive integer value."

    trials_dict = {}

    for d, i in itertools.product(d_range, range(1, n_runs + 1)):
        kmeans = MiniBatchKMeans(
            n_clusters = n_clusters,
            batch_size = batch_size,
            random_state = random_state
        )

        # cluster the points
        kmeans.fit(data[:, :d])

        # add results into dictionary
        trials_dict[rk(d, i)] = rv(kmeans.cluster_centers_, kmeans.labels_, kmeans.inertia_)
    
    return trials_dict


def combine_facilities(dict_object, K, n_facility, batch_size, random_state):
    """
    Combine facilities in a dictionary object.
    """
    from sklearn.cluster import MiniBatchKMeans
    import numpy as np

    # check that the dictionary object is formatted correctly
    assert isinstance(dict_object, dict)
    for key, value in dict_object.items():
        assert isinstance(key, rk)
        assert isinstance(value, rv)

    dict_object_combined = {}

    # combine the facilities from different trials
    for key, value in dict_object.items():
        facilities = value.facility
        labels = value.labels

        # count the number of cells assigned to each facility
        # facilities with no cells are dropped
        #indices, weights = np.unique(labels, return_counts = True)
        weights = list(map(
            lambda cluster_no: np.count_nonzero(labels == cluster_no), 
            range(0, n_facility)
        ))

        # cluster the facility using weighted k means
        kmeans = MiniBatchKMeans(
            n_clusters = K,
            batch_size = batch_size,
            random_state = random_state
        )
        kmeans.fit(facilities, sample_weight = weights)

        # write result
        dict_object_combined[key] = rv(kmeans.cluster_centers_, kmeans.labels_[labels], kmeans.inertia_)

    return dict_object_combined



def convert_dict_into_binary_matrix(dict_object, true_n_clusters, true_n_cells):
    """ 
    Convert clustering runs in a dictionary into a binary consensus matrix.
    """

    import numpy as np
    import logging

    # check the number of cells
    n_cells_list = [len(x.labels) for x in dict_object.values()]
    assert n_cells_list.count(true_n_cells) == len(n_cells_list), "number of cells not consistent"
    n_cells = true_n_cells

    # check the number of clusters
    # there is an edge case for this, which is if the kth cluster is not assigned any cells
    # will fix if error occurs, otherwise can just count that at least 75% of runs is correct?
    n_clusters_list = [np.max(x.labels) + 1 for x in dict_object.values()]
    assert n_clusters_list.count(true_n_clusters) == len(n_clusters_list), "number of cells not consistent"
    n_clusters =  true_n_clusters

    # initialise empty B array with the correct shape
    B = np.zeros((n_cells, 0), dtype=int)

    # for each run, we create the block to be appended to B
    for i, value in enumerate(dict_object.values()):
        cell_labels = value.labels

        # create a total of 'n_clusters' columns for this iteration
        b = np.zeros((n_cells, n_clusters), dtype=int)

        # annotate the correct columns for each row/cell
        b[range(0, n_cells), cell_labels] = 1

        # this checks that every cell only has one cluster assignment
        assert np.all(np.sum(b, axis=1) == 1), "some cells have multiple cluster assignments"

        # append to the right of the overall binary consensus matrix
        B = np.append(B, b, axis=1)

    # remove columns in B with no cells assigned 
    # this happens if some initialised centroids do not get assigned anything
    logging.info(f'original binary matrix shape: {B.shape}')
    cols_with_no_cells = np.sum(B, axis=0) != 0
    if not np.all(cols_with_no_cells):
        B = B[:, np.sum(B, axis=0) != 0]
        logging.warning(f'remove some clusters with no assigned cells, binary matrix shape is now: {B.shape}')  

    return B


def cluster_binary_matrix(binary_consensus_matrix, n_clusters, batch_size, random_state):
    """
    Cluster the consensus binary matrix.
    """

    from sklearn.cluster import MiniBatchKMeans

    kmeans = MiniBatchKMeans(
        n_clusters = n_clusters,
        batch_size = batch_size,
        random_state = random_state
    )
    kmeans.fit(binary_consensus_matrix)

    return kmeans.labels_