from ._matrix import svd_scipy, svd_sklearn
import numpy as np
from scipy.cluster.vq import kmeans2 as kmeans
from scipy.sparse import issparse
from sklearn.cluster import MiniBatchKMeans

def strm_spectral_new(data,
                  k = 100, 
                  lowrankdim = 25, # rename as n_component
                  stream = 1000,
                  batch = 1000,
                  n_runs = 5,
                  svd = "sklearn",
                  randomcellorder = True):
    """
    Cluster the cells into microclusters. k should be set to a relatively large number, maybe a 
    fraction of the first batch?

    In the code, `centroids` obviously refers to microcentroids.

    To do: incorporate parallel functionality using the same Laplacian.

    Need to modularise steps more (see separate notes).
    """
    
    # initialise and print parameters
    i = 0
    j = stream
    n_cells, n_genes = data.shape
    print(f"""PARAMETERS:
    
    num_cells: {n_cells}, n_genes: {n_genes}
    
    stream size: {stream}
    number of components: {lowrankdim}
    SVD algorithm: {svd}

    """)

    # generate look up table for the random ordering of cells
    lut = np.arange(n_cells) 
    if randomcellorder is True:
        np.random.shuffle(lut)

    # create dictionary with centroids and assignments of the k-means runs
    runs = {t: {'cent': np.empty((0, lowrankdim)),
                'asgn': np.empty(n_cells, dtype='int32')} for t in range(0, n_runs)}

    # initialise structures to be updated during training
    sketch_matrix = np.empty((lowrankdim, n_genes)) # learned embedding
    gene_sum = np.empty(n_genes) # to calculate dataset centroid
    Vold = None # to skip initial rotation
    
    print("beginning the stream...\n")
    while i < n_cells:
        # obtain range of current stream
        if (n_cells - i) < stream: # resize last stream
            j = n_cells - i 
            sketch_matrix = sketch_matrix[:(lowrankdim + j), :] # i think not needed
    
            def resize_centroids(run, k, j):
                run['cent'] = run['cent'][:(k+j)]
                return run
            runs = {t: resize_centroids(run, k, j) for t, run in runs.items()}
        else:
            j = stream

        # obtain current stream
        if issparse(data[lut[i:(i+j)], ]):
            cells = data[lut[i:(i+j)], ].todense()
        else:
            cells = data[lut[i:(i+j)], ]
        
        # update embedding
        sketch_matrix, V, u, gene_sum = _embed(cells, C, gene_sum, lowrankdim, svd)

        # rotate centroids
        Vold = V if Vold is None
        #assert np.all(Vold != V), "rotation matrices are the same!"
        runs = {t: _rotate_centroids(run, V, Vold) for t, run in runs.items()}

        # assign new cells to centroids, centroids are reinitialised by random chance
        runs = {t: assign(run, u, k, lut, i, j) for t, run in runs.items()}

        print(f"working on observations {i} to {i+j}...")
        i += j
        Vold = V

    # unrandomise the ordering
    runs = {(lowrankdim, t): _reorder_cells(run, lut) for t, run in runs.items()}

    return runs

def _embed(cells, sketch_matrix, sum_gene, lowrankdim, svd = "sklearn"):
    assert len(sum_gene.shape) == 1
    assert cells.shape[0] == sum_gene.shape[0]
    assert sketch_matrix.shape[0] == lowrankdim

    stream_size = cells.shape[0]

    # calculate the sum_vector based on the new cells
    sum_gene = sum_gene + np.sum(cells, axis = 0)

    # calculate dataset centroid up to current point in stream
    dataset_centroid = sum_gene / np.sum(sum_gene)

    # approximate degree matrix (n_cell x n_cell)
    D = np.diag(np.dot(cells, dataset_centroid))
    assert np.shape == (stream_size, stream_size), "degree matrix incorrect size"

    # approximated normalised Laplacian
    laplacian = np.dot(np.sqrt(D), cells)

    # concatenate with previous sketch matrix and perform SVD
    C = np.empty(lowrankdim + stream_size, sketch_matrix.shape[1])
    C[:lowrankdim] = sketch_matrix
    C[lowrankdim:] = laplacian

    # choose svd algorithm (maybe use a switch case)
    # https://jaxenter.com/implement-switch-case-statement-python-138315.html
    if svd == "sklearn":
        U, s, V = svd_sklearn(C, lowrankdim)
    elif svd == "scipy":
        U, s, V = svd_scipy(C, lowrankdim)
    
    # update sketch matrix
    s2 = s**2
    s_norm = np.sqrt(s2 - s2[-1])
    sketch_matrix = np.dot(np.diag(s_norm), V)

    # normalise the landmarks in each cell (row-wise)
    u = U / np.transpose(np.tile(np.linalg.norm(U, axis=1), (lowrankdim, 1)))

    # (return the whole U as it is, which is a n_cell x k column, subset as necessary)
    return sketch_matrix, V, u, sum_genes
    

 

def _assign(run, cell_projections, k, lut, i, j, batch = 100, restart_chance = 0.05):
    assert run['cent'].shape[1] == cell_projections.shape[1], 
        "dimensions of centroids and cell projections not equal"
    assert cell_projections.shape[0] == i + j + 1, 
        "LUT indices and number of cells not equal"
    # should also test that the run contains cent and assgn (can do in arguments?)
    # add random state later on

    # join current cells with previous points to be clustered together
    n_centroid = run['cent'].shape[0]
    points = np.concatenate((run['cent'], cell_projections), axis=0)

    # initialise k means object, to add: random state
    # by random chance, centroids are reinitialised
    if np.random.rand() < restart_chance:
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

        # WHAT DOES THIS DO AGAIN??
           def resize_centroids(run, k, j):
                run['cent'] = run['cent'][:(k+j)]
                return run
            runs = {t: resize_centroids(run, k, j) for t, run in runs.items()}



def _strm_kmeans(centroids, current_cells, k, assignments, i, restartchance = 0.05):
    """
    Cluster the rows in `points` into `k` clusters, using the first `k` points
    as initial cluster centers, which are then also updated.
    
    Add `assignments` for new points, and also updates those of the previous `i` points.

    `points` and `assignments` must have the same depth (i.e. number of realisations).
    """
    points = np.concatenate((centroids, current_cells), axis=0)

    # assign new cells to centroids
    # by random chance, we reinitialise the centroids
    if np.random.rand() < restartchance:
        points, new_assignments = kmeans(points, k, iter=1000, thresh=1e-5, minit="random")
        print("reinitialised centroids!\n")
    else:
        points, new_assignments = kmeans(points, points[:k,], iter=500, thresh=1e-5, minit="matrix")
    
    centroids = points[:k]

    # update assignments for the previous cells, based on the updated centroids
    assignments[:i] = new_assignments[assignments[:i]]

    # write assignments for current cells
    assignments[i:] = new_assignments[k:,]

    return centroids, assignments



def _cluster_subsequent_cells(run, current_cells, k, lut, i, j):
    """
    Executes streaming k-means in parallel. `run` is a dictionary holding the result of an individual run.
    """
    run['cent'], run['asgn'][lut[:(i+j)]] = _strm_kmeans(run['cent'], 
        current_cells, k, assignments = run['asgn'][lut[:(i+j)]], i = i)
    return run


def _cluster_initial_cells(run, current_cells, k, lut, index):
    """
    Clusters the first cells in parallel. `run` is a dictionary holding the result of an individual run.
    """
    run['cent'], run['asgn'][lut[:index]] = kmeans(current_cells, k, iter=500, thresh=1e-5)
    return run




runs = {t: assign(run, u, k, lut, i, j) for t, run in runs.items()}

def _rotate_centroids(run, V, Vold):
    """
    Rotate centroids from landmark into old gene space, then into updated landmark space.
    """
    run['cent'] = np.dot(np.dot(run['cent'], Vold), np.transpose(V))
    return run

def _reorder_cells(run, lut):
    """
    Reorder the cells given the lookup table.
    """
    assignments = run['asgn']
    reordered_assignments = np.empty(len(assignments), dtype=int)
    reordered_assignments[lut]= assignments[lut]
    return run



