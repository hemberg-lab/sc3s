from ._misc import _check_integer_single

def convert_dict_into_contigency_matrix(dict_object, true_n_clusters, true_n_cells):
    """
    Convert clustering runs in a dictionary into a contingency consensus matrix.
    """
    
    import numpy as np

    n_trials = len(dict_object.values())

    # check the number of cells
    n_cells_list = [len(x['labels']) for x in dict_object.values()]
    assert n_cells_list.count(true_n_cells) == len(n_cells_list), "number of cells not consistent"
    n_cells = true_n_cells

    # check the number of clusters
    # there is an edge case for this, which is if the kth cluster is not assigned any cells
    # will fix if error occurs, otherwise can just count that at least 75% of runs is correct?
    n_clusters_list = [np.max(x['labels']) + 1 for x in dict_object.values()]
    assert n_clusters_list.count(true_n_clusters) > 0.9 * len(n_clusters_list), "number of cells not consistent"
    n_clusters =  true_n_clusters

    # initialise contingeny matrix
    C = np.zeros((n_cells, n_cells))

    # obtain the clustering labels from each trial as a matrix
    cell_labels_matrix = np.array([trial['labels'] for trial in dict_object.values()]).T
    assert cell_labels_matrix.shape == (n_cells, n_trials)

    # count the number of matches for each cell
    for cell in range(0, n_cells):
        C[cell, ] = np.sum(cell_labels_matrix[cell, :] == cell_labels_matrix, axis = 1) / n_trials

    # this might be expensive
    assert np.allclose(C, C.T), "something went wrong, contingency matrix not symmetrical"
    
    return C


def cluster_matrix_kmeans(consensus_matrix, n_clusters, random_state):
    """
    Wrapper around scikit learn whole batch k means. Rows of matrix should be observations.
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

    # check random state
    from sklearn.utils import check_random_state
    random_state = check_random_state(random_state)

    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    kmeans.fit(consensus_matrix)

    return kmeans.labels_

def cluster_matrix_agglomerative(consensus_matrix, n_clusters):
    """
    Wrapper around scikit learn agglomerative clustering. Rows of matrix should be observations.
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

    from sklearn.cluster import AgglomerativeClustering
    agg_clust = AgglomerativeClustering(n_clusters=n_clusters)
    agg_clust.fit(consensus_matrix)

    return agg_clust.labels_