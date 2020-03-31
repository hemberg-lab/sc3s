import numpy as np
import pandas as pd
import scanpy as sc
import math
from scipy import linalg
from sklearn.decomposition import TruncatedSVD

def svd_scipy(X, n_components):
    """
    Singular value decomposition using `scipy.linalg.svd`.
    Returned matrices are truncated to the value of `n_components`.

    Parameters
    ----------
    X : TYPE
        DESCRIPTION.
    n_components : TYPE
        DESCRIPTION.

    Returns
    -------
    U : TYPE
        DESCRIPTION.
    s : TYPE
        DESCRIPTION.
    Vh : TYPE
        DESCRIPTION.

    """
    U, s, Vh = linalg.svd(X, full_matrices=False)
    U  = U[:, :n_components]
    s  = s[:n_components]
    Vh = Vh[:n_components, ]
    return U, s, Vh

def inv_svd(U, s, Vh):
    """
    Inverse of the singular value decomposition.

    Parameters
    ----------
    U : TYPE
        DESCRIPTION.
    s : TYPE
        DESCRIPTION.
    Vh : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    return np.dot(U, np.dot(np.diag(s), Vh))

def svd_sklearn(X, n_components, n_iter=5, random_state=None):
    """
    Truncated singular value decomposition using `scikitlearn`.

    Parameters
    ----------
    X : TYPE
        DESCRIPTION.
    n_components : TYPE
        DESCRIPTION.
    n_iter : TYPE, optional
        DESCRIPTION. The default is 5.
    random_state : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    U : TYPE
        DESCRIPTION.
    s : TYPE
        DESCRIPTION.
    Vh : TYPE
        DESCRIPTION.

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

def streamSC(data, k = 3, batchmode = False, svd_algorithm = "sklearn",
             initial = 0.2, stream = 0.02, lowrankdim = 0.05, iterations = 5,
             initialmin = 10**3, streammin = 10,
             initialmax = 10**5, streammax = 100):
    
    # initialmin is a hard limit below which we use batch clustering
    # it could be eventually hard coded into the algorithm
    # streammin can also be hard coded
    # we probably also do the same for their maximum
    assert initialmin <= initialmax
    assert streammin <= streammax
    
    # randomise ordering of cells
    # expr_matrix = np.transpose(data.X) 
    
    # obtain and calculate useful values
    # there's quite some edge cases that I need to think about for this part
    n_cells, n_genes = data.X.shape

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
    
    # choose the SVD solver
    # need to factor in random seed
    if svd_algorithm == "sklearn":
        svd = svd_sklearn
    elif svd_algorithm == "scipy":
        svd = svd_scipy
        
    # SVD of the first, initial large batch
    i, j = 0, initial
    print("clustering the initial batch ...")
    print(i, j)
    U, s, Vh = svd(np.transpose(data.X[i:(i+j),]), lowrankdim)
    
    # initialise empty array for "previous batch"
    # row: genes, cell: low rank dim
    # B = np.empty((n_genes, lowrankdim))
    # U = [] # rotation matrix (don't think this is needed, this is not Julia)
    
    # normalise the sigma singular values (step 7 of strmEMB)
    s2 = s**2
    s_norm = np.sqrt(s2 - s2[-1])
    
    # scale the columns in U with normalised singular values
    B = np.dot(U, np.diag(s_norm))
    print(B.shape)

    # for consensus results
    # these is for nParallele executions of B    
    #compM1 = zeros(nParallel, lowRankCells+1, maxK);
    
    # centroids
    # to come
    
    print("... initial batch finished!\n")
    i += initial
    
    # initialise empty array for previous batch
    # this is to prevent using concatenate (memory hungry) --- to benchmark
    C = np.empty((n_genes, lowrankdim + stream))
    print(C.shape)
    
    # do the streaming
    print("streaming the remaining cells ...")
    while i < n_cells:
        # obtain range of current stream
        if (n_cells - i) < stream:
            j = n_cells - i  # last stream may not be exact size
            C = C[:, :(lowrankdim + j)] # resize
        else:
            j = stream
        
        # obtain current stream and concatenate to previous batch
        current_cells = np.transpose(data.X[i:(i+j), ])
        C[:, :lowrankdim] = B # landmarks from previous stream
        C[:, lowrankdim: ] = current_cells
        
        # svd operations
        U, s, Vh = svd(C, lowrankdim)
        s2 = s**2
        s_norm = np.sqrt(s2 - s2[-1])
        B = np.dot(U, np.diag(s_norm))

        # for consensus results
        # these is for nParallele executions of B    
        #compM1 = zeros(nParallel, lowRankCells+1, maxK);
        
        print(i, i + j, current_cells.shape, U.shape)
        i += j
        
    # unrandomise the ordering
    print("... stream finished!\n")
    
    return 0
