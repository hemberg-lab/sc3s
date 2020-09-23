from ._matrix import _svd_sklearn
from ._misc import _parse_int_list
import numpy as np
from scipy.cluster.vq import kmeans2 as kmeans
from scipy.sparse import issparse
from sklearn.cluster import MiniBatchKMeans

def _spectral(data,
              k = 100, 
              d_range = 25,
              stream = 1000,
              batch = 100,
              n_runs = 5,
              svd = "sklearn",
              randomcellorder = True,
              return_centers = True,
              restart_chance = 0.05):
    """
    Cluster the cells into microclusters. k should be set to a relatively large number, maybe a 
    fraction of the first batch?

    In the code, `centroids` obviously refers to microcentroids.

    To do: incorporate parallel functionality using the same Laplacian.

    Need to modularise steps more (see separate notes).
    """

    n_cells, n_genes = data.shape

    # check parameters
    assert isinstance(k, int)
    assert isinstance(stream, int) and stream <= n_cells
    assert isinstance(batch, int) and batch <= stream
    assert isinstance(n_runs, int) and n_runs >= 1
    assert isinstance(randomcellorder, bool)
    assert 0 <= restart_chance <= 1
    d_range = _parse_int_list(d_range)
    print(f"running spectral clustering, num_components = {d_range}")

    i, j = 0, stream
    
    # look up table for randomising cell order
    lut = np.arange(n_cells) 
    if randomcellorder is True:
        np.random.shuffle(lut)

    # create dictionary with centroids and assignments of the k-means runs
    runs = {}
    for d in d_range:
        for t in range(n_runs):
            runs[(t,d)] = {'cent': np.empty((0, d)), 'asgn': np.empty(n_cells, dtype='int32')} 

    # initialise structures to be updated during training
    max_d = max(d_range)
    sketch_matrix = np.empty((max_d, n_genes)) # learned embedding
    gene_sum = np.zeros(n_genes) # to calculate dataset centroid
    Vold = None # to skip initial rotation

    while i < n_cells:
        # obtain range of current stream
        if (n_cells - i) < stream: # resize last stream
            j = n_cells
        else:
            j = i + stream

        print(f"working on observations {i} to {j}...")

        # obtain current stream
        if isinstance(data, np.ndarray):
            cells = data[lut[i:j], ]
        else:
            cells = data[lut[i:j], ].toarray()
        
        # update embedding
        sketch_matrix, V, U, gene_sum = _embed(cells, sketch_matrix, gene_sum, max_d, svd)

        # rotate centroids
        Vold = V if Vold is None else Vold

        # run the clustering at different values of d
        for d in d_range:
            # extract the correct number of components, normalise the landmarks by cell (row-wise)
            cell_projections = U[:, :d]
            cell_projections /= np.transpose(np.tile(np.linalg.norm(cell_projections, axis=1), (d, 1)))

            # extract relevant V
            V_d    = V[:d,]
            Vold_d = V[:d,]

            for t in range(n_runs):
                runs[(t,d)] = _rotate_centroids(runs[(t,d)], V_d, Vold_d)
                runs[(t,d)] = _assign(runs[(t,d)], cell_projections, k, lut, i, j, batch, restart_chance)

        # update index for next stream
        i = j
        Vold = V

    # unrandomise the ordering
    runs = {k: _reorder_cells(run, lut) for k, run in runs.items()}

    # return only the cell assignments if requested
    if return_centers is False:
        runs = {k: v['asgn'] for k, v in runs_dict.items()}

    print(f"...done!\n")

    return runs

def _embed(cells, sketch_matrix, gene_sum, max_d, svd = "sklearn"):
    assert len(gene_sum.shape) == 1
    assert cells.shape[1] == gene_sum.shape[0], "not the same number of genes/features"
    assert sketch_matrix.shape[0] == max_d

    stream_size = cells.shape[0]

    # calculate the sum_vector based on the new cells
    gene_sum = gene_sum + np.sum(cells, axis = 0)

    # calculate dataset centroid up to current point in stream
    dataset_centroid = gene_sum / np.linalg.norm(gene_sum) # gene_sum.shape[0]

    # approximate degree matrix (n_cell x n_cell)
    D = np.diag(np.dot(cells, dataset_centroid))
    assert D.shape == (stream_size, stream_size), "degree matrix incorrect size"

    # approximated normalised Laplacian
    laplacian = np.dot(np.sqrt(D), cells)

    # concatenate with previous sketch matrix and perform SVD
    C = np.empty((max_d + stream_size, sketch_matrix.shape[1]))
    C[:max_d] = sketch_matrix
    C[max_d:] = laplacian

    # choose svd algorithm (maybe use a switch case)
    # https://jaxenter.com/implement-switch-case-statement-python-138315.html
    U, s, V = _svd_sklearn(C, max_d)
    
    # update sketch matrix
    s2 = s**2
    s_norm = np.sqrt(s2 - s2[-1])
    sketch_matrix = np.dot(np.diag(s_norm), V)

    # extract cell projections (n_cell x d column)
    cell_projections = U[max_d:, ]

    return sketch_matrix, V, cell_projections, gene_sum


def _assign(run, cell_projections, k, lut, i, j, batch = 100, restart_chance = 0.05):
    assert run['cent'].shape[1] == cell_projections.shape[1], "dimensions of centroids and cell projections not equal"
    assert cell_projections.shape[0] == (j - i), "LUT indices and number of cells not equal"
    # should also test that the run contains cent and assgn (can do in arguments?)
    # add random state later on

    # join current cells with previous points to be clustered together
    n_centroid = run['cent'].shape[0]
    points = np.concatenate((run['cent'], cell_projections), axis=0)

    # initialise k means object, to add: random state
    # by random chance, centroids are reinitialised
    if (np.random.rand() < restart_chance) or (n_centroid == 0):
        kmeans = MiniBatchKMeans(n_clusters=k, batch_size=batch) 
    else:
        kmeans = MiniBatchKMeans(n_clusters=k, batch_size=batch, init=run["cent"]) 
    
    # run k means function and extract results
    kmeans = kmeans.fit(points)
    run['cent'] = kmeans.cluster_centers_
    centroid_reassignments, cell_assignments = kmeans.labels_[:n_centroid], kmeans.labels_[n_centroid:]
    
    # updates assignments for previous cells based on new centroid labels
    run['asgn'][lut[:i]] = centroid_reassignments[run['asgn'][lut[:i]]]

    # write assignments for cells in current stream
    run['asgn'][lut[i:j]] = cell_assignments
    
    return run

def _rotate_centroids(run, V, Vold):
    """
    Rotate centroids from landmark into old gene space, then into updated landmark space.
    """
    centroids = run['cent']
    assert V.shape == Vold.shape
    assert centroids.shape[1] == V.shape[0]
    run['cent'] = np.dot(np.dot(centroids, Vold), np.transpose(V))
    return run

def _reorder_cells(run, lut):
    """
    Reorder the cells given the lookup table.
    """
    assignments = run['asgn']
    reordered_assignments = np.empty(len(assignments), dtype=int)
    reordered_assignments[lut]= assignments[lut]
    return run



