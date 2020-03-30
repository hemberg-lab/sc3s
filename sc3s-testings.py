import numpy as np
import pandas as pd
import scanpy as sc
import math

sc.settings.verbosity = 3
sc.logging.print_versions()
sc.settings.set_figure_params(dpi=80)

adata = sc.read_10x_mtx(
    './data/filtered_gene_bc_matrices/hg19/',
    var_names='gene_symbols', cache=True)
adata.var_names_make_unique()
adata

adata.isbacked

# basic preprocessing
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)

# gene names (variables)
adata.var_names

# filter by mitochondrial proportion
# add that a columns in the obs dataframe
mito_genes = adata.var_names.str.startswith('MT-')
adata.obs['percent_mito'] = np.sum(
    adata[:, mito_genes].X, axis=1) / np.sum(adata.X, axis=1)
adata.obs['n_counts'] = adata.X.sum(axis=1)
adata = adata[adata.obs.n_genes < 2500, :]
adata = adata[adata.obs.percent_mito < 0.05, :]

# normalise by library size so it's 10,000 reads per cell
sc.pp.normalize_total(adata, target_sum=1e4)

# log transform
sc.pp.log1p(adata)

# store as the raw attribute
# don't need this is dont do regress_out and scale
adata.raw = adata

# identify high variable genes
sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
adata.var.columns # we have new columns added
sc.pl.highly_variable_genes(adata)
adata = adata[:, adata.var.highly_variable]

# regress out effects of total counts per cell and percentage of mito genes
sc.pp.regress_out(adata, ['n_counts', 'percent_mito'])

# write results
# adata.write(results_file)
adata

##########

# LEIDEN METHOD (default in scanpy)
# compute neighbourhood graph
sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
sc.tl.umap(adata)
sc.tl.leiden(adata) # clustering
sc.pl.umap(adata, color='leiden')

# results_file = "./write/pbmc3k.h5ad"
# adata.write(results_file)

#########

# CONSENSUS CLUSTERING METHOD

# the function should be a mutating function
# returns an anndata object
# clusters assigned as a new columns in the .obs dataframe
# information about the run parameters as a dictionary in .uns

adata.X.shape

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


calculate_rmse(inv_svd(A[0], A[1], A[2]), adata.X)
calculate_rmse(inv_svd(B[0], B[1], B[2]), adata.X)

l = round(0.2 * adata.X.shape[1])
A = svd_scipy(adata.X, l)
B = svd_sklearn(adata.X, l)

inv_svd(A[0], A[1], A[2])
inv_svd(B[0], B[1], B[2])


svd_random = TruncatedSVD(l, algorithm="randomized", random_state=None)
U2  = svd_random.fit_transform(adata.X) # this has already been scaled by the singular vectors
Vh2 = svd_random.components_ 
s2  = svd_random.singular_values_ 
U2_uns = U2 / np.tile(s2, (U2.shape[0], 1))
reconstructed2 = svd_random.inverse_transform(U2) # you need to keep the svd_random object
reconstructed2_manual = np.dot(U2, Vh2)
    
    

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
        svd = svd_sklearn(X, lowrankdim, n_iter=5, random_state=None)
    elif svd_algorithm == "scipy":
        svd_scipy(X, lowrankdim)
        
    
    # initialise empty array for "previous batch"
    # row: genes, cell: low rank dim
    B = np.empty((n_genes, lowrankdim))
    #U = [] # rotation matrix (don't think this is needed, this is not Julia)
    
    # for consensus results
    # these is for nParallele executions of B    
    #compM1 = zeros(nParallel, lowRankCells+1, maxK);
    
    # centroids
    # to come
    
    # do the streaming
    i = 0
    print("beginning stream...")
    while i < n_cells:
        
        # obtain range of current stream
        if i == 0:
            j = initial # the initial batch
        elif (n_cells - i) < stream:
            j = n_cells - i  # last stream may not be exact size
        else:
            j = stream
        
        # obtain current batch, concatenate to previous batch and increment counter
        current_cells = adata.X[i:(i+j), ]
        print(i, i + j, current_cells.shape)
        
        # should consider putting the first batch outside, because the size is so different?
        # would also make it easier to benchmark later on
        if i == 0:
            
        # need deepcopy?
        
        i += j
        
        # get U and B from previous time step (need to deepcopy?)
        if i
        
        """
        # get U and B from previous step
        Uold = deepcopy(U); Bold = deepcopy(B);

        # append the new stream, otherwise use the first batch 
        # compM1 is denoted as C_t in the paper
        if ~isempty(B)
            compM1 = convert(Array{Float64, 2}, hcat(B, current_cells)); #step5 of alg5
        else
            compM1 = current_cells;
        end
        
        # perform the SVD
        tmp = svd(compM1);
        U = tmp.U[:,1:lowRankCells];
        S = tmp.S[1:lowRankCells];
        V = tmp.V[:,1:lowRankCells];

        # normalise the sigma singular values (step 7 of strmEMB)
        delta = S[lowRankCells]^2;
        Sn = diagm(0 => max.((S.^2 .- delta), 0).^(1/2));
        Bfull = real(U * Sn);
        """
        
    # unrandomise the ordering
    print("stream finished...")

#######################################################
# testing svd methods

adata.X.shape

import seaborn as sns
import matplotlib.pyplot as plt

from scipy import linalg

# low rank dimension
l = round(0.5 * adata.X.shape[1])

U, s, Vh = linalg.svd(adata.X, full_matrices=True)
U.shape,  s.shape, Vh.shape # (n_cells, n_genes), (n_genes,), (n_genes, n_genes)

reconstructed_full = np.dot(U, np.dot(np.diag(s), Vh))
reconstructed_less = np.dot(U[:,:l], np.dot(np.diag(s[:l]), Vh[:l,]))

# check similarity
np.allclose(adata.X, reconstructed_full, atol = 1e-3)
error_full = reconstructed_full - adata.X
np.sum(error_full ** 2) # rmse
sns.distplot(error_full)

np.allclose(adata.X, reconstructed_less, atol = 1)
error_less = reconstructed_less - adata.X
np.sum((reconstructed_less - adata.X) ** 2) # rmse
sns.distplot(error_less)

# i can't get this subplot to work for some reason
f, axes = plt.subplots(2, 1, figsize=(7, 7), sharex=True)
sns.despine(left=True)
sns.distplot(error_full, color="b", ax=axes[0, 0])
sns.distplot(reconstructed_less - adata.X, hist=False, rug=True, color="r", ax=axes[1, 0])
plt.setp(axes, yticks=[])
plt.tight_layout()



##########

# NOW to test the truncated SVD
# http://blog.explainmydata.com/2016/01/how-much-faster-is-truncated-svd.html


# Truncated SVD with scikit-learn
from sklearn.decomposition import TruncatedSVD

# randomised version
l = round(0.2 * adata.X.shape[1])
svd_random = TruncatedSVD(l, algorithm="randomized", random_state=None)
U2  = svd_random.fit_transform(adata.X) # this has already been scaled by the singular vectors
Vh2 = svd_random.components_ 
s2  = svd_random.singular_values_ 
U2_uns = U2 / np.tile(s2, (U2.shape[0], 1))
reconstructed2 = svd_random.inverse_transform(U2) # you need to keep the svd_random object
reconstructed2_manual = np.dot(U2, Vh2)

np.round(U2/U[:,0:l])[:,0:5]
np.round(Vh2/Vh[0:l],1)[0:5,]
s2[0:5]

np.round(U2 / U[:,0:l])[:,-4:]
np.round(Vh2/Vh[0:l],1)[-4:,]
s2[-4:]

# middle bit
m = l//4 *3
np.round(U2 / U[:,0:l])[:,m:(m+5)]
np.round(Vh2/Vh[0:l],1)[m:(m+5),]
s2[m:(m+5)]


error2 = reconstructed2 - adata.X
np.sum(error2 ** 2) # rmse
sns.distplot(error2)


# arpack version
svd_arpack = TruncatedSVD(l, algorithm="arpack", random_state=None, tol=0.0)
U3  = svd_arpack.fit_transform(adata.X)
Vh3 = svd_arpack.components_
s3  = svd_arpack.singular_values_ 
reconstructed3 = svd_random.inverse_transform(U3) # you need to keep the svd_random object


error3 = reconstructed3 - adata.X
np.sum(error3 ** 2) # rmse
sns.distplot(error3)


# compare images
import plotly.io as pio
pio.renderers.default = "browser"
import plotly.express as px
px.imshow(adata.X)
px.imshow(reconstructed_less)
px.imshow(reconstructed2)
px.imshow(reconstructed2_manual)
px.imshow(reconstructed3)


def np_svd(X):
    return np.linalg.svd(X, full_matrices=False, compute_uv=True)

def np_inv_svd(svd_outputs):
    U, s, V = svd_outputs
    return np.dot(U, np.dot(np.diag(s), V))

def sklearn_randomized_svd(X, rank=RANK):
    tsvd = sklearn.decomposition.TruncatedSVD(rank, algorithm="randomized", n_iter=1)
    X_reduced = tsvd.fit_transform(X)
    return (tsvd, X_reduced)

def sklearn_arpack_svd(X, rank=RANK):
    tsvd = sklearn.decomposition.TruncatedSVD(rank, algorithm="arpack")
    X_reduced = tsvd.fit_transform(X)
    return (tsvd, X_reduced)

def sklearn_inv_svd(svd_outputs):
    tsvd, X_reduced = svd_outputs
    return tsvd.inverse_transform(X_reduced)

def decompose_matrix()