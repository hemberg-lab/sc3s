from ._utils import svd_scipy, svd_sklearn
import numpy as np
import math
from scipy.cluster.vq import kmeans2 as kmeans

def strm_spectral(data, k = 100, streammode = True, svd_algorithm = "sklearn",
                  initial = 0.2, stream = 0.02, lowrankdim = 0.05, n_parallel = 5,
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
        
    # obtain and calculate useful values
    # there's quite some edge cases that I need to think about for this part
    n_cells, n_genes = data.shape

    # look up table for the ordering of cells in the stream
    lut = np.arange(n_cells) 
    if randomcellorder is True:
        np.random.shuffle(lut)

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
    
    # choose the SVD solver
    # need to factor in random seed
    if svd_algorithm == "sklearn":
        svd = svd_sklearn
    elif svd_algorithm == "scipy":
        svd = svd_scipy

    # print all the parameters
    print(f"PARAMETERS:\n\nnum_cells: {n_cells}, n_genes: {n_genes}\n\ninitial: {initial}\nstream: {stream}\nnumber of landmarks: {lowrankdim}\nsvd: {svd_algorithm}\n")
        
    # initialise parameters
    i, j = 0, initial
    centroids = np.empty((k + stream, lowrankdim))
    assignments = np.empty(n_cells, dtype='int32')
    C = np.empty((lowrankdim + stream, n_genes)) # array storing embeddings and new cells

    print(f"clustering the initial batch, observations {i} to {j}...")
    C[:lowrankdim, ], V, current_cells = _create_embedding(data[lut[i:(i+j)],], lowrankdim, svd)
    print("... initial batch finished!\n")

    # cluster the first batch and obtain centroids
    # u_first is not needed afterwards
    centroids[:k,], assignments[lut[i:(i+j)]] = kmeans(current_cells, k, iter=500, thresh=1e-5)

    i += initial
    
    # do the streaming
    print("now streaming the remaining cells ...\n")
    while i < n_cells:
        # obtain range of current stream
        if (n_cells - i) < stream: # last stream may not be exact and need to be resized
            j = n_cells - i 
            C = C[:(lowrankdim + j), :] 
            centroids = centroids[:(k+j)]
        else:
            j = stream
        
        # obtain current stream and concatenate to previous memory
        C[lowrankdim:, :] = data[lut[i:(i+j)], ]

        # update embedding and rotate centroids from previous iteration
        C, V, centroids[:k,], centroids[k:,] = _update_embedding(C, V, centroids[:k,], centroids[k:,], lowrankdim, svd)

        # for consensus results
        # these is for nParallel executions of B
        #compM1 = zeros(nParallel, lowRankCells+1, maxK);

        # assign new cells to centroids, centroids are reinitialised by random chance
        centroids, assignments[lut[:(i+j)]] = _strm_kmeans(centroids, k, assignments[lut[:(i+j)]], i)

        print(f"working on observations {i} to {i+j}...")
        i += j

    # final microcentroids, removing the last stream
    centroids = centroids[:k]
        
    # unrandomise the ordering
    reordered_assignments = np.empty(n_cells, dtype=int)
    reordered_assignments[lut]= assignments[lut]
    print("\nspectral clustering finished!")
    
    return centroids, reordered_assignments

def _create_embedding(C, lowrankdim, svd):
    """
    Initialise the spectral embedding of the low rank matrix.
    * `C`: gene coordinates of the first batch of cells.
    * `lowrankdim`: eigenvectors to keep in svd (i.e. how much to remember).
    It represents the number of memorable (meta)cells to keep for next iteration.
    * `svd`: SVD algorithm to use.

    Returns:
    * `V`: rotation from gene into updated landmark space.
    * `c`: gene coordinates of updated memorable points.
    * `u`: coordinates of the new cells in updated landmark space (normalised `U`).
    """
    # singular value decomposition
    U, s, V = svd(C, lowrankdim)

    # update coordinates of the most memorable points
    # i.e. first `lowrankdim` rows of C
    s2 = s**2
    s_norm = np.sqrt(s2 - s2[-1])
    c = np.dot(np.diag(s_norm), V) # spectral embedding
    
    # normalise the landmarks in each cell (row-wise)
    u = U / np.transpose(np.tile(np.linalg.norm(U, axis=1), (lowrankdim, 1)))

    return c, V, u

def _update_embedding(C, V, centroids, points, lowrankdim, svd):
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
    # rotate centroids from landmark into gene space
    centroids = np.dot(centroids, V)

    # singular value decomposition
    U, s, V = svd(C, lowrankdim)

    # update coordinates of the most memorable points
    # i.e. first `lowrankdim` rows of C
    s2 = s**2
    s_norm = np.sqrt(s2 - s2[-1])
    C[:lowrankdim, :] = np.dot(np.diag(s_norm), V)

    # rotate centroids from gene into updated landmark space
    centroids = np.dot(centroids, np.transpose(V))
    
    # get landmark coordinates of new cells
    # normalise the landmarks in each cell (by row), since not all singular vectors kept
    points = U[lowrankdim:,]
    points = points / np.transpose(np.tile(np.linalg.norm(points, axis=1), (lowrankdim, 1)))

    return C, V, centroids, points


def _strm_kmeans(points, k, assignments, i):
    """
    Cluster the rows in `points` into `k` clusters, using the first `k` points
    as initial cluster centers, which are then also updated.
    
    Add `assignments` for new points, and also updates those of the previous `i` points.

    `points` and `assignments` must have the same depth (i.e. number of realisations).

    Chance to restart is 0.05.
    """
    # assign new cells to centroids
    # by random chance, we reinitialise the centroids
    if np.random.rand() < 0.05:
        points[:k,], new_assignments = kmeans(points, k, iter=500, thresh=1e-5, minit="random")
        print("reinitialised centroids!\n")
    else:
        points[:k,], new_assignments = kmeans(points, points[:k,], iter=100, thresh=1e-5, minit="matrix")
    
    # update assignments for the previous cells
    assignments[:i] = new_assignments[assignments[:i]]
    assignments[i:] = new_assignments[k:,]

    return points, assignments