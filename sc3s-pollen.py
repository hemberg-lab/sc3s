#!/usr/bin/env python3

import numpy as np
import pandas as pd

dfexp = pd.read_table("data/pollen2-exp.txt", header=None)
dflab = pd.read_table("data/pollen2-labels.txt", header=None)

expMatrix = np.transpose(dfexp.iloc[:, 1:dfexp.shape[1]]) # row: gene, column: cell
celltypes = np.transpose(dflab)

import scanpy as sc
import anndata as an

a =  expMatrix.to_numpy()
adata = sc.AnnData(expMatrix.to_numpy())

adata.obs = celltypes

sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
sc.tl.umap(adata)
sc.tl.leiden(adata) # clustering
sc.pl.umap(adata, color='leiden')



# test the new algorithm
A = generate_microclusters(adata, k=100, initial = 100, stream = 50, lowrankdim = 20)
A = generate_microclusters(adata, k=100, initial = 100, stream = 10, lowrankdim = 20)
A = generate_microclusters(adata, k=100, initial = 100, stream = 10, lowrankdim = 20)

A = generate_microclusters(adata, k=25, initial=25, stream = 20, lowrankdim = 20)
A = generate_microclusters(adata, k=25, initial=25, stream = 5, lowrankdim = 20) # one cluster is very unevenly sized

# now consolidate microclusters into clusters
macrocentroids, macroclusters = kmeans(A[0], 4)
adata.obs['sc3'] = macroclusters[A[1]]
macroclusters[A[1]]

#adata.obs['sc3'] = A[1]
sc.pl.umap(adata, color='leiden', size=35)
sc.pl.umap(adata, color='sc3', size=35)

sc.tl.pca(adata, svd_solver='arpack')
sc.pl.pca(adata, color='leiden', size=35)
sc.pl.pca(adata, color='sc3', size=35)