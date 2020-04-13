import numpy as np
import pandas as pd
import scanpy as sc
import math
from scipy import linalg
from sklearn.decomposition import TruncatedSVD

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

###########

from scipy.cluster.vq import kmeans2 as kmeans
kmeans(X, initX, iter=100, thresh=1e-5, minit="matrix")


A = generate_microclusters(adata, 8)
adata.obs['sc3'] = A[1]
sc.pl.umap(adata, color='sc3')

#streamSC(adata, 5)

generate_microclusters(adata, 5)


sc3s.streamSC(adata, 5)

B = streamSC(adata, 5)
B.shape

sc3s.inv_svd(B[0], B[1], B[2]).shape # not all the cells, but only the last batch






l = round(0.2 * adata.X.shape[1])
A = sc3s.svd_scipy(adata.X, l)
B = sc3s.svd_sklearn(adata.X, l)

sc3s.calculate_rmse(sc3s.inv_svd(A[0], A[1], A[2]), adata.X)
sc3s.calculate_rmse(sc3s.inv_svd(B[0], B[1], B[2]), adata.X)

sc3s.inv_svd(A[0], A[1], A[2])
sc3s.inv_svd(B[0], B[1], B[2])


svd_random = TruncatedSVD(l, algorithm="randomized", random_state=None)
U2  = svd_random.fit_transform(adata.X) # this has already been scaled by the singular vectors
Vh2 = svd_random.components_ 
s2  = svd_random.singular_values_ 
U2_uns = U2 / np.tile(s2, (U2.shape[0], 1))
reconstructed2 = svd_random.inverse_transform(U2) # you need to keep the svd_random object
reconstructed2_manual = np.dot(U2, Vh2)
    
    


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



