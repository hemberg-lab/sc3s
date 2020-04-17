import numpy as np
import pandas as pd
import scanpy as sc
import math
from scipy import linalg
from sklearn.decomposition import TruncatedSVD
from scipy.cluster.vq import kmeans2 as kmeans

def svd_scipy(X, n_components):
    """
    Singular value decomposition using `scipy.linalg.svd`.
    Returned matrices are truncated to the value of `n_components`.
    """
    U, s, Vh = linalg.svd(X, full_matrices=False)
    U  = U[:, :n_components]
    s  = s[:n_components]
    Vh = Vh[:n_components, ]
    return U, s, Vh

def inv_svd(U, s, Vh):
    """
    Inverse of the singular value decomposition.
    """
    return np.dot(U, np.dot(np.diag(s), Vh))

def svd_sklearn(X, n_components, n_iter=5, random_state=None):
    """
    Truncated singular value decomposition using `scikitlearn`.
    """
    svd = TruncatedSVD(n_components, algorithm="randomized", n_iter=n_iter,
                       random_state=random_state)
    U = svd.fit_transform(X)
    s = svd.singular_values_
    Vh = svd.components_
    U = U / np.tile(s, (U.shape[0],1)) # by default, U is scaled by s
    return U, s, Vh

def calculate_rmse(A, B):
    error = A - B
    return np.sum(error ** 2)

def generate_microclusters(data, k = 100, batchmode = False, svd_algorithm = "sklearn",
                           initial = 0.2, stream = 0.02, lowrankdim = 0.05, iterations = 5,
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
    n_cells, n_genes = data.X.shape

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
        
    # SVD of the first, initial large batch
    i, j = 0, initial
    print("clustering the initial batch ...")
    print(i, j)
    U, s, Vh = svd(data.X[lut[i:(i+j)],], lowrankdim)
    print(U.shape, s.shape, Vh.shape)
    
    # initialise empty arrays for "previous batch" (B) and previous V
    # row: genes, cell: low rank dim
    # (don't think this is needed, this is not Julia)
    
    # normalise the sigma singular values (step 7 of strmEMB)
    s2 = s**2
    s_norm = np.sqrt(s2 - s2[-1])
    
    # scale the columns in U with normalised singular values
    B = np.dot(np.diag(s_norm), Vh) # np.dot(U, np.diag(s_norm)) #change
    print("B: ", B.shape)

    # for consensus results (to come)
    # these is for nParallele executions of B    
    #compM1 = zeros(nParallel, lowRankCells+1, maxK);
    
    # prepare empty arrays for centroid coordinates and cell assignments
    centroids = np.empty((k, lowrankdim))
    assignments = np.empty(n_cells, dtype='int32')
    
    # cluster the first batch and obtain centroids
    centroids, assignments[lut[i:(i+j)]] = kmeans(U, k, iter=500, thresh=1e-5)
    
    print("... initial batch finished!\n")
    i += initial
    
    # initialise empty array for previous batch
    # this is to prevent using concatenate (memory hungry) --- to benchmark
    C = np.empty((lowrankdim + stream, n_genes)) # np.empty((n_genes, lowrankdim + stream))
    print("C: ", C.shape)
    
    # do the streaming
    print("streaming the remaining cells ...")
    while i < n_cells:
        # obtain range of current stream
        if (n_cells - i) < stream: # last stream may not be exact and need to be resized
            j = n_cells - i 
            C = C[:(lowrankdim + j), :] # C[:, :(lowrankdim + j)]
        else:
            j = stream
        
        # obtain current stream and concatenate to previous batch
        current_cells = data.X[lut[i:(i+j)], ]
        C[:lowrankdim, :] = B # landmarks from previous stream
        C[lowrankdim:, :] = current_cells

        # svd operations
        Vh_old = Vh
        U, s, Vh = svd(C, lowrankdim)
        s2 = s**2
        s_norm = np.sqrt(s2 - s2[-1])
        B = np.dot(np.diag(s_norm), Vh) # B = np.dot(U, np.diag(s_norm))

        # rotate centroids into the new landmark space (using the right singular vectors)
        centroids = np.dot(centroids, np.dot(Vh_old, np.transpose(Vh)))
        
        # normalise the current cells
        new_u = U[lowrankdim:,]
        norms = np.transpose(np.tile(np.linalg.norm(new_u, axis=1), (lowrankdim, 1)))
        new_u = new_u / norms # normalise the landmarks in each cell (by row)

        combined = np.concatenate((centroids, new_u), axis=0)

        # for consensus results
        # these is for nParallel executions of B    
        #compM1 = zeros(nParallel, lowRankCells+1, maxK);

        # assign new cells to centroids
        # by random chance, we reinitialise the centroids
        if np.random.rand() < 0.05:
            centroids, new_assignments = kmeans(combined, k, iter=500, thresh=1e-5, minit="random")
            print("reinitialised centroids!")
        else:
            centroids, new_assignments = kmeans(combined, centroids, iter=100, thresh=1e-5, minit="matrix")
        
        # update assignments for the previous cells
        assignments[lut[:i]] = new_assignments[assignments[lut[:i]]]
        assignments[lut[i:(i+j)]] = new_assignments[k:,]

        print(i, i+j, assignments[lut[0:15]])

        """
        print(i, i + j, 'current cells: ', current_cells.shape, 
            'U: ', U[lowrankdim:,].shape, 
            'centroids: ', centroids.shape, 
            'newly assigned: ', new_assignments[k:, ].shape)
        """
        i += j
        
    # unrandomise the ordering
    reordered_assignments = np.empty(n_cells, dtype=int)
    reordered_assignments[lut]= assignments[lut]
    print("... stream finished!\n")
    
    return centroids, reordered_assignments