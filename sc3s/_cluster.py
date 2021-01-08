from ._misc import rk, rv
from ._misc import _check_and_format_integer_list, _check_integer_single, _check_dict_object

def run_trials_miniBatchKMeans(data, n_clusters, d_range, n_runs, batch_size, random_state):
    """
    Generate dictionary object containing MiniBatchKMeans clustering runs.
    """

    import itertools
    from sklearn.cluster import MiniBatchKMeans

    # check that consensus matrix has no NaN, infinity values
    from sklearn.utils import check_array
    data = check_array(data)

    # check n_clusters
    n_clusters = _check_integer_single(
        n_clusters,
        min_val = 2,
        var_name = 'n_clusters'
    )

    # check d_range
    d_range = _check_and_format_integer_list(
        d_range,
        min_val = 2,
        max_val = data.shape[1],
        var_name = 'd_range'
    )

    # check n_runs
    n_runs = _check_integer_single(n_runs, min_val=1, var_name="n_runs")

    # check batch_size
    batch_size = _check_integer_single(
        batch_size,
        min_val = 10,
        max_val = data.shape[0],
        var_name = 'batch_size'
    )

    # check random state
    from sklearn.utils import check_random_state
    random_state = check_random_state(random_state)

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
        # trials_dict[rk(d, i)] = rv(  # NEED SCANPY TO IMPLEMENT SAVING
        trials_dict[(d, i)] = rv(
            facility = kmeans.cluster_centers_,
            labels = kmeans.labels_,
            inertia = kmeans.inertia_
        )
    
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
        # assert isinstance(key, rk)  # NEED SCANPY TO IMPLEMENT SAVING
        assert isinstance(value, dict)
        assert value['facility'] is not None
        assert value['labels'] is not None

    # check arguments are formated as integers (limits not trivial to do)
    K = _check_integer_single(K, var_name = 'K')
    n_facility = _check_integer_single(n_facility, var_name = 'n_facility')
    batch_size = _check_integer_single(batch_size, var_name = 'batch_size')

    # check random state
    from sklearn.utils import check_random_state
    random_state = check_random_state(random_state)

    dict_object_combined = {}

    # combine the facilities from different trials
    for key, value in dict_object.items():
        facilities = value['facility']
        labels = value['labels']

        # count the number of cells assigned to each facility
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
        dict_object_combined[key] = rv(
            facility = kmeans.cluster_centers_,
            labels = kmeans.labels_[labels], 
            inertia = kmeans.inertia_
        )

    return dict_object_combined



def convert_dict_into_binary_matrix(dict_object, true_n_clusters, true_n_cells):
    """ 
    Convert clustering runs in a dictionary into a binary consensus matrix.
    """

    import numpy as np
    import logging

    n_clusters, n_cells = true_n_clusters, true_n_cells

    # check dict_object has correct number of clusters and cells for every run
    dict_object = _check_dict_object(dict_object, true_n_clusters, true_n_cells)

    # initialise empty B array with the correct shape
    B = np.zeros((n_cells, 0), dtype=int)

    # for each run, we create the block to be appended to B
    for i, value in enumerate(dict_object.values()):
        cell_labels = value['labels']

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


def cluster_consensus_matrix(consensus_matrix, n_clusters, batch_size, random_state):
    """
    Cluster the consensus binary matrix using miniBatchKmeans. Rows of matrix should be observations.
    """

    # check that consensus matrix has no NaN, infinity values
    from sklearn.utils import check_array
    consensus_matrix = check_array(consensus_matrix)

    # check n_clusters
    n_clusters = _check_integer_single(
        n_clusters,
        min_val = 2,
        max_val = consensus_matrix.shape[0],
        var_name = 'n_clusters'
    )

    # check batch_size
    batch_size = _check_integer_single(
        batch_size,
        min_val = 10,
        max_val = consensus_matrix.shape[0],
        var_name = 'batch_size'
    )

    # check random state
    from sklearn.utils import check_random_state
    random_state = check_random_state(random_state)

    from sklearn.cluster import MiniBatchKMeans
    kmeans = MiniBatchKMeans(
        n_clusters = n_clusters,
        batch_size = batch_size,
        random_state = random_state
    )
    kmeans.fit(consensus_matrix)

    return kmeans.labels_