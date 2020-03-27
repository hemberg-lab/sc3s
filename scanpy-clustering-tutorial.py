# https://icb-anndata.readthedocs-hosted.com/en/stable/fileformat-prose.html

# !mkdir data
# !wget http://cf.10xgenomics.com/samples/cell-exp/1.1.0/pbmc3k/pbmc3k_filtered_gene_bc_matrices.tar.gz -O data/pbmc3k_filtered_gene_bc_matrices.tar.gz
# !cd data; tar -xzf pbmc3k_filtered_gene_bc_matrices.tar.gz

import numpy as np
import pandas as pd
import scanpy as sc

sc.settings.verbosity = 3             # verbosity: errors (0), warnings (1), info (2), hints (3)
sc.logging.print_versions()
sc.settings.set_figure_params(dpi=80)

results_file = "./write/pbmc3k.h5ad"

adata = sc.read_10x_mtx(
    './data/filtered_gene_bc_matrices/hg19/',
    var_names='gene_symbols', cache=True)
adata.var_names_make_unique()
adata

adata.isbacked

##########

# PREPROCESSING

sc.pl.highest_expr_genes(adata, n_top=20, )

# basic filtering
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)

# gene names (variables)
adata.var_names

# calculate proportion of mitochondrial genes
# add that a columns in the obs dataframe
mito_genes = adata.var_names.str.startswith('MT-')
adata.obs['percent_mito'] = np.sum(
    adata[:, mito_genes].X, axis=1) / np.sum(adata.X, axis=1)
adata.obs['n_counts'] = adata.X.sum(axis=1)

# violin plot of these
sc.pl.violin(adata, ['n_genes', 'n_counts', 'percent_mito'],
             jitter=0.4, multi_panel=True)

# remove cells with too many mitochondrial genes or too many total counts
sc.pl.scatter(adata, x='n_counts', y='percent_mito')
sc.pl.scatter(adata, x='n_counts', y='n_genes')

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

# pca
sc.tl.pca(adata, svd_solver='arpack')
sc.pl.pca(adata, color='CST3')
sc.pl.pca_variance_ratio(adata, log=True) #scree plot

# write results
adata.write(results_file)
adata

##########

# NEIGHBOURHOOD GRAPHS

# compute neighbourhood graph
sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)

# UMAP
sc.tl.umap(adata)
sc.pl.umap(adata, color=['CST3', 'NKG7', 'PPBP']) # color be gene expression

# you can also plot the raw uncorrected gene expression
# (it's still normalised and log tansformed though)
sc.pl.umap(adata, color=['CST3', 'NKG7', 'PPBP'], use_raw=False)

# Leiden clusters based on the community detection method
# which was already computed by the way
sc.tl.leiden(adata)
sc.pl.umap(adata, color='leiden')

# write results into the same file?!
adata.write(results_file)


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
