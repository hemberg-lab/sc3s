import numpy as np
import pandas as pd
import scanpy as sc

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

import math

def streamSC(data, k = 3, batchmode = False, initial = 0.2,
             stream = 0.02, lowrankdim = 0.05, iterations = 5,
             initialmin = 10**3, streammin = 10,
             initialmax = 10**5, streammax = 100):
    
    # initialmin is a hard limit below which we use batch clustering
    # it could be eventually hard coded into the algorithm
    # streammin can also be hard coded
    # we probably also do the same for their maximum
    assert initialmin <= initialmax
    assert streammin <= streammax
    
    # randomise ordering of cells
    
    
    
    # obtain and calculate useful values
    # there's quite some edge cases that I need to think about for this part
    n_cells, n_genes = data.X.shape

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
        
        # obtain current batch and increment counter
        print(i, i + j)
        i += j
        
    # unrandomise the ordering
    print("stream finished...")


adata.X.shape
from scipy import linalg
U, s, Vh = linalg.svd(adata.X, full_matrices=False)
U.shape,  s.shape, Vh.shape
reconstructed = np.dot(U, np.dot(np.diag(s), Vh))

l = round(0.5 * adata.X.shape[1])# low rank
reconstructed_less = np.dot(U[:,:l], np.dot(np.diag(s[:l]), Vh[:l,]))

# check similarity
np.allclose(adata.X, reconstructed, atol = 1e-3)
np.sum((reconstructed - adata.X) ** 2) # rmse

np.allclose(adata.X, reconstructed_less, atol = 1)
np.sum((reconstructed_less - adata.X) ** 2) # rmse

import plotly.io as pio
pio.renderers.default = "browser"
import plotly.express as px
from plotly.subplots import make_subplots

px.imshow(adata.X)
px.imshow(reconstructed)
px.imshow(reconstructed_less)

#fig = make_subplots(rows = 1, cols = 3)
#fig.append_trace(px.imshow(reconstructed_less), row=1, col=1) #doesn't work
#fig

# NOW to test the truncated SVD




################################################################################
################################################################################


adata.obs['consensus'] = adata.obs['leiden']
test_palette = ['black'] + adata.uns['leiden_colors'][1:]

# nice! very easy to customise
sc.pl.umap(adata, color='consensus', palette=test_palette)


##########

# FINDING MARKER GENES

# compute a ranking for the highly DE genes in each cluster
# done with the t-test
sc.tl.rank_genes_groups(adata, 'leiden', method='t-test')
sc.pl.rank_genes_groups(adata, n_genes=25, sharey=False)

# alternative is the Wilcoxon rank-sum test (recommended)
sc.tl.rank_genes_groups(adata, 'leiden', method='wilcoxon')
sc.pl.rank_genes_groups(adata, n_genes=25, sharey=False)
adata.write(results_file)

# we can also use more unconventional multivariate tests
# like a logistic regression
sc.tl.rank_genes_groups(adata, 'leiden', method='logreg')
sc.pl.rank_genes_groups(adata, n_genes=25, sharey=False)

# thus we decide that:
marker_genes = ['IL7R', 'CD79A', 'MS4A1', 'CD8A', 'CD8B', 'LYZ', 'CD14',
                'LGALS3', 'S100A8', 'GNLY', 'NKG7', 'KLRB1',
                'FCGR3A', 'MS4A7', 'FCER1A', 'CST3', 'PPBP']


# reload the object saved with the Wilcoxon rank-sum test
adata = sc.read(results_file)

# show the top 10 ranked genes per cluster 0,1,...,7
pd.DataFrame(adata.uns['rank_genes_groups']['names']).head(5)

result = adata.uns['rank_genes_groups']
groups = result['names'].dtype.names
pd.DataFrame(
    {group + '_' + key[:1]: result[key][group]
     for group in groups for key in ['names', 'pvals']}).head(5)

# compare to a single cluster (here we're comparing clusters 2 and 3 with 1)
sc.tl.rank_genes_groups(adata, 'leiden', groups=['2','3'], reference='1', method='wilcoxon')
sc.pl.rank_genes_groups(adata, groups=['2', '3'], n_genes=20)

# violin plots
sc.pl.rank_genes_groups_violin(adata, groups=['2','3'], n_genes=8)

# compare a gene across groups
sc.pl.violin(adata, ['CST3', 'NKG7', 'PPBP'], groupby='leiden')

# actually mark the cell types
new_cluster_names = [
    'CD4 T', 'CD14 Monocytes',
    'B', 'CD8 T',
    'NK', 'FCGR3A Monocytes',
    'Dendritic', 'Megakaryocytes', 'why is there an extra cluster']
adata.rename_categories('leiden', new_cluster_names)

sc.pl.umap(adata, color='leiden', legend_loc='on_data', title='', frameon=False)

# visualise the marker genes
ax = sc.pl.dotplot(adata, marker_genes, groupby='leiden')
ax = sc.pl.stacked_violin(adata, marker_genes, groupby='leiden', rotation=90)

# save the annotations
adata
adata.write(results_file, compression="gzip")

# you can also save the file without the dense and scaled data matrix
# the sparse matrix is saved as .raw
adata.X = None
adata.write('./write/pbmc3k_withoutX.h5ad', compression='gzip')

