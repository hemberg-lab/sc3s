#!/usr/bin/env python3

import numpy as np
import pandas as pd
import scanpy as sc

dfexp = pd.read_table("data/pollen2-exp.txt", header=None)
dflab1 = pd.read_table("data/pollen1-labels.txt", header=None)
dflab2 = pd.read_table("data/pollen2-labels.txt", header=None)

expMatrix = np.transpose(dfexp.iloc[:, 1:dfexp.shape[1]]) # row: gene, column: cell

adata = sc.AnnData(expMatrix.to_numpy())
adata.obs = pd.DataFrame({'label1': pd.Series(dflab1.iloc[0], dtype='category'), 'label2': pd.Series(dflab2.iloc[0], dtype='category')})

sc.pp.filter_genes(adata, min_cells=50)
sc.pp.log1p(adata)

# calculate neighborhood map
sc.pp.neighbors(adata, n_neighbors=20, n_pcs=40, use_rep='X')
sc.tl.umap(adata)
sc.tl.leiden(adata)
sc.tl.pca(adata, n_comps=10, svd_solver='arpack')

# test the new algorithm
A = generate_microclusters(adata, k=100, initial = 100, stream = 50, lowrankdim = 20)
A = generate_microclusters(adata, k=100, initial = 100, stream = 10, lowrankdim = 20)

A = generate_microclusters(adata, k=25, initial=25, stream = 20, lowrankdim = 20)
A = generate_microclusters(adata, k=25, initial=25, stream = 5, lowrankdim = 20) # one cluster is very unevenly sized

# now consolidate microclusters into clusters
# this is the scipy version
A = generate_microclusters(adata, k=200, initial = 100, stream = 10, lowrankdim = 20)
macrocentroids, macroclusters = kmeans(A[0], 4)
adata.obs = adata.obs.assign(sc3_k4 = pd.Categorical(macroclusters[A[1]]))
adata.obs['sc3_k4']

adata.obs = adata.obs.assign(sc3_k11 = pd.Categorical(macroclusters[A[1]]))
adata.obs['sc3_k11']


# I'm using the scikit version here, because it supports weighted k means
from sklearn.cluster import KMeans
for _ in range(0,10):
    A = generate_microclusters(adata, k=200, initial = 100, stream = 10, lowrankdim = 20)
    
    (uniq_mclst, count_mclst) = np.unique(A[1], return_counts = True)
    weights = np.zeros(A[0].shape[0], dtype=int)
    weights[uniq_mclst] = count_mclst

    kmeans_weight = KMeans(n_clusters=4).fit(A[0], sample_weight=weights)
    macroclusters = kmeans_weight.labels_
    macrocentroids = kmeans_weight.cluster_centers_
    adata.obs = adata.obs.assign(sc3_k4 = pd.Categorical(macroclusters[A[1]]))
    sc.pl.pca(adata, color='sc3_k4', size=35)

# UMAP and PCA plots
sc.pl.umap(adata, color='leiden', size=35)

sc.pl.umap(adata, color='label1', size=35)
sc.pl.umap(adata, color='sc3_k11', size=35)

sc.pl.umap(adata, color='label2', size=35)
sc.pl.umap(adata, color='sc3_k4', size=35)

sc.pl.pca_variance_ratio(adata, log=True)
sc.pl.pca(adata, color='label1', size=35)
sc.pl.pca(adata, color='sc3_k11', size=35)
sc.pl.pca(adata, color='label2', size=35)
sc.pl.pca(adata, color='sc3_k4', size=35)

# testing how to randomise cells
S = 5
letters = np.array(list(map(lambda x: chr(ord('a') + x), range(0,S))), dtype='str')
arr = np.arange(S)
np.random.shuffle(arr)
arr

letters[arr]

adata.obs[arr, :]


