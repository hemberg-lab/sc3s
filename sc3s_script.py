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
        s_norm = s # s_norm = np.sqrt(s2 - s2[-1])
        B = np.dot(U, np.diag(s_norm))

        # for consensus results
        # these is for nParallel executions of B    
        #compM1 = zeros(nParallel, lowRankCells+1, maxK);
        
        print(i, i + j, current_cells.shape, U.shape)
        i += j
        
    # unrandomise the ordering
    print("... stream finished!\n")
    
    return U, s, Vh


def generate_microclusters(data, k = 5, batchmode = False, svd_algorithm = "sklearn",
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

    # print all the parameters
    print(f"PARAMETERS:\n\nnum_cells: {n_cells}, n_genes: {n_genes}\n\ninitial: {initial}\nstream: {stream}\nnumber of landmarks: {lowrankdim}\nsvd: {svd_algorithm}\n")
        
    # SVD of the first, initial large batch
    i, j = 0, initial
    print("clustering the initial batch ...")
    print(i, j)
    U, s, Vh = svd(data.X[i:(i+j),], lowrankdim) # svd(np.transpose(data.X[i:(i+j),]), lowrankdim) 
    
    # initialise empty array for "previous batch"
    # row: genes, cell: low rank dim
    # B = np.empty((n_genes, lowrankdim))
    # U = [] # rotation matrix (don't think this is needed, this is not Julia)
    
    # normalise the sigma singular values (step 7 of strmEMB)
    s2 = s**2
    s_norm = np.sqrt(s2 - s2[-1])
    
    # scale the columns in U with normalised singular values
    B = np.dot(np.diag(s_norm), Vh) # np.dot(U, np.diag(s_norm)) #change
    print("B: ", B.shape)

    # for consensus results
    # these is for nParallele executions of B    
    #compM1 = zeros(nParallel, lowRankCells+1, maxK);
    
    # cluster the first batch and obtain centroids
        # initialise array first I think TO DO!!!!!!!!
    centroids, assignments = kmeans(U, k, iter=100, thresh=1e-5)

    # to come
    
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
        current_cells = data.X[i:(i+j), ] # np.transpose(data.X[i:(i+j), ])
        # C[:, :lowrankdim] = B # landmarks from previous stream
        # C[:, lowrankdim: ] = current_cells
        C[:lowrankdim, :] = B # landmarks from previous stream
        C[lowrankdim:, : ] = current_cells

        # svd operations
        U, s, Vh = svd(C, lowrankdim)
        s2 = s**2
        s_norm = s # s_norm = np.sqrt(s2 - s2[-1])
        B = np.dot(np.diag(s_norm), Vh) # B = np.dot(U, np.diag(s_norm))

        

        # for consensus results
        # these is for nParallel executions of B    
        #compM1 = zeros(nParallel, lowRankCells+1, maxK);
        combined = np.concatenate((centroids, U), axis=0)
        centroids, assignments = kmeans(combined, 
            centroids, iter=100, thresh=1e-5, minit="matrix")
        
        # rotate centroids using the right singular vectors

        print(i, i + j, current_cells.shape, Vh.shape, combined.shape)
        i += j
        
    # unrandomise the ordering
    print("... stream finished!\n")
    
    return centroids, U #U, s, Vh



def streaming_kmeans(x, centroids):
    # stream the x points into the centroids, maintaining the same number of centroids
    centroids, assignments = kmeans(x, centroids, iter=100, thresh=1e-5, minit="matrix")
    pass



# a = streamSC(adata, 5)
# b = generate_microclusters(adata, 5)

# a[0].shape, a[1].shape, a[2].shape
# b[0].shape, b[1].shape, b[2].shape

# a_re = np.dot(a[0], np.dot(np.diag(a[1]), a[2]))

# b_re = np.dot(b[0], np.dot(np.diag(b[1]), b[2]))
