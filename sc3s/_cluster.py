from ._utils import svd_scipy, svd_sklearn, weighted_kmeans
import numpy as np
import math
import itertools
from scipy.cluster.vq import kmeans2 as kmeans
from scipy.sparse import issparse

def strm_spectral(data, k = 100, 
                  streammode = True, svd_algorithm = "sklearn",
                  initial = 0, stream = 0, lowrankdim = 0.05, n_parallel = 5,
                  initialmin = 10**3, streammin = 10,
                  initialmax = 10**5, streammax = 100, randomcellorder = True):
    """
    Cluster the cells into microclusters. k should be set to a relatively large number, maybe a 
    fraction of the first batch?

    In the code, `centroids` obviously refers to microcentroids.

    To do: incorporate parallel functionality using the same Laplacian.

    Need to modularise steps more (see separate notes).
    """
    
    # initialmin is a hard limit below which we use batch clustering
    # it could be eventually hard coded into the algorithm
    # streammin can also be hard coded
    # we probably also do the same for their maximum
    assert initialmin <= initialmax
    assert streammin <= streammax

    """
    # calculate low rank representation size
    # we want the representation of cells in a lower dimensionality space, not of genes
    # I actually think rounding up is a mistake, should round down and throw an error if needed
    lowrankdim = math.ceil(lowrankdim * n_cells)

    # calculate size of initial stream
    if n_cells < initialmin + 1:
        initial = n_cells
    else:
        initial = int(math.ceil(initial * n_cells / 10) * 10) # nearest 10th
        if initial > n_cells: # can this happen in practice?
            initial = n_cells
            
    # calculate size of subsequent streams
    if initial != n_cells:
        stream = int(math.ceil(stream * n_cells / 10) * 10) # nearest 10th
        if stream < streammin + 1:
            stream = streammin
        elif stream > streammax:
            stream = streammax
    """
    
    # initialise, and print all the parameters
    i, j = 0, initial
    n_cells, n_genes = data.shape

    if svd_algorithm == "sklearn":
        svd = svd_sklearn
    elif svd_algorithm == "scipy":
        svd = svd_scipy

    print(f"PARAMETERS:\n\nnum_cells: {n_cells}, n_genes: {n_genes}\n\ninitial: {initial}\nstream: {stream}\nnumber of landmarks: {lowrankdim}\nsvd: {svd_algorithm}\n")

    # look up table for the ordering of cells in the stream
    lut = np.arange(n_cells) 
    if randomcellorder is True:
        np.random.shuffle(lut)

    # dict with centroids and assignments of several k-means initialisations
    runs = {t: {'cent': np.empty((k, lowrankdim)),
                'asgn': np.empty(n_cells, dtype='int32')} for t in range(0, n_parallel)}
    
    print(f"clustering the initial batch, observations {i} to {j}...")
    C = np.empty((lowrankdim + stream, n_genes)) # array storing embeddings and current cells
    C[:lowrankdim, ], V, current_cells = _create_embedding(data[lut[i:(i+j)],], lowrankdim, svd)
    print("... initial batch finished!\n")

    # cluster the first batch and obtain centroids
    runs = {t: _cluster_initial_cells(run, current_cells, k, lut, i+j) for t, run in runs.items()}

    # do the rest of the stream
    print("now streaming the remaining cells ...\n")
    Vold = V
    i += initial
    while i < n_cells:
        # obtain range of current stream
        if (n_cells - i) < stream: # last stream may not be exact and need to be resized (wrap into a function)
            j = n_cells - i 
            C = C[:(lowrankdim + j), :] 

            def resize_centroids(run, k, j):
                run['cent'] = run['cent'][:(k+j)]
                return run
            runs = {t: resize_centroids(run, k, j) for t, run in runs.items()}
        else:
            j = stream
        
        # obtain current stream and concatenate to previous memory
        if issparse(data[lut[i:(i+j)], ]):
            C[lowrankdim:, :] = data[lut[i:(i+j)], ].todense()
        else:
            C[lowrankdim:, :] = data[lut[i:(i+j)], ]

        # update embedding and rotate centroids from previous iteration
        C, V, current_cells = _update_embedding(C, lowrankdim, svd)

        # rotate centroids
        assert np.all(Vold != V), "rotation matrices are the same!"
        runs = {t: _rotate_centroids(run, V, Vold) for t, run in runs.items()}

        # assign new cells to centroids, centroids are reinitialised by random chance
        runs = {t: _cluster_subsequent_cells(run, current_cells, k, lut, i, j) for t, run in runs.items()}

        print(f"working on observations {i} to {i+j}...")
        i += j
        Vold = V
       
    # unrandomise the ordering
    runs = {(lowrankdim, t): _reorder_cells(run, lut) for t, run in runs.items()}

    # consolidate microclusters into macroclusters, add the lowrankdim in the key
    #runs = {(lowrankdim, t): weighted_kmeans(run['cent'], run['asgn'], num_clust) for t, run in runs.items()}
    print("\nspectral clustering finished!\n")

    return runs

def _create_embedding(cells_first, lowrankdim, svd):
    """
    Initialise the spectral embedding of the low rank matrix.
    * `cells_first`: gene coordinates of the first batch of cells.
    * `lowrankdim`: eigenvectors to keep in svd (i.e. how much to remember).
    It represents the number of memorable (meta)cells to keep for next iteration.
    * `svd`: SVD algorithm to use.

    Returns:
    * `V`: rotation from gene into updated landmark space.
    * `c`: gene coordinates of updated memorable points.
    * `u`: coordinates of the new cells in updated landmark space (normalised `U`).
    """
    # singular value decomposition
    U, s, V = svd(cells_first, lowrankdim)

    # update coordinates of the most memorable points
    # i.e. first `lowrankdim` rows of cells_first
    s2 = s**2
    s_norm = np.sqrt(s2 - s2[-1])
    C = np.dot(np.diag(s_norm), V)  # spectral embedding
    
    # normalise the landmarks in each cell (row-wise)
    u = U / np.transpose(np.tile(np.linalg.norm(U, axis=1), (lowrankdim, 1)))

    return C, V, u

def _update_embedding(C, lowrankdim, svd):
    """
    Update the spectral embedding of the low rank matrix and rotate the centroids.
    * `C`: gene coordinates of previous memorable points and new data.
    Function updates the first `lowrankdim` rows as the most memorable 
    information from previous iteration (i.e. metacells), based on
    the remaining rows (i.e. incoming stream of new cells).
    * `V`: rotation from landmark into gene space.
    * `centroids`: coordinates of the maintained centroids in landmark space.
    * `points`: coordinates of the previous cells in the old landmark space
    (this is not needed for learning, it's simply to write those of the next cells).
    * `lowrankdim`: eigenvectors to keep in svd (i.e. how much to remember).
    It represents the number of memorable (meta)cells to keep for next iteration.
    * `svd`: SVD algorithm to use.

    I'm considering spliting the `C` argument into two.
    Or keep it as it is, but consider the new data as a separate argument.

    Landmark: metagene.

    REALLY worth having unit tests to make sure the mutation works!

    This is a mutating functions, and updates these inputs:
    * `C`: gene coordinates of updated memorable points, and the untouched (new) cells.
    * `V`: rotation from gene into updated landmark space.
    * `centroids`: coordinates of the maintained centroids in updated landmark space.
    * `points`: coordinates of the new cells in updated landmark space.
    """
    # singular value decomposition
    U, s, V = svd(C, lowrankdim)

    # update coordinates of the most memorable points
    # i.e. first `lowrankdim` rows of C
    s2 = s**2
    s_norm = np.sqrt(s2 - s2[-1])
    C[:lowrankdim, :] = np.dot(np.diag(s_norm), V)

    # get landmark coordinates of new cells
    # normalise the landmarks in each cell (by row), since not all singular vectors kept
    current_cells = U[lowrankdim:,]
    current_cells = current_cells / np.transpose(np.tile(np.linalg.norm(current_cells, axis=1), (lowrankdim, 1)))

    return C, V, current_cells


def _rotate_centroids(run, V, Vold):
    """
    Rotate centroids from landmark into old gene space, then into updated landmark space.
    """
    run['cent'] = np.dot(np.dot(run['cent'], Vold), np.transpose(V))
    return run


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


def _reorder_cells(run, lut):
    """
    Reorder the cells given the lookup table.
    """
    assignments = run['asgn']
    reordered_assignments = np.empty(len(assignments), dtype=int)
    reordered_assignments[lut]= assignments[lut]
    return run
