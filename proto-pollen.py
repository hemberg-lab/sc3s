#!/usr/bin/env python3

cd /Users/fq1/code/sc3s

import sc3s

import scanpy as sc
import pandas as pd
import numpy as np

dfexp = pd.read_table("data/pollen2-exp.txt", header=None)
dflab1 = pd.read_table("data/pollen1-labels.txt", header=None)
dflab2 = pd.read_table("data/pollen2-labels.txt", header=None)

expMatrix = np.transpose(dfexp.iloc[:, 1:dfexp.shape[1]]) # row: gene, column: cell

adata = sc.AnnData(expMatrix.to_numpy())
adata.obs = pd.DataFrame({'label1': pd.Series(dflab1.iloc[0], dtype='category'), 'label2': pd.Series(dflab2.iloc[0], dtype='category')})

# preprocessing to remove uninformative genes, and log transform
sc.pp.filter_genes(adata, min_cells=50)
sc.pp.log1p(adata)
adata

# calculate neighborhood map, and perform Leiden clustering
sc.pp.neighbors(adata, n_neighbors=20, n_pcs=40, use_rep='X')
sc.tl.umap(adata)
sc.tl.leiden(adata)
sc.tl.pca(adata, n_comps=10, svd_solver='arpack')

##################################################

# SC3s 
# generate microclusters
microcentroids, assignments = sc3s.tl.strm_spectral(adata, k=100, initial = 100, stream = 20, lowrankdim = 20)

# now consolidate microclusters into clusters
cell_assignments = sc3s.tl.weighted_kmeans(microcentroids, assignments)
adata.obs = adata.obs.assign(sc3_k4_single = pd.Categorical(cell_assignments))

# UMAP plots
sc.pl.umap(adata, color='label2', size=35)
sc.pl.umap(adata, color='sc3_k4_single', size=35)

# PCA plots
sc.pl.pca_variance_ratio(adata, log=True)
sc.pl.pca(adata, color='label2', size=35)
sc.pl.pca(adata, color='sc3_k4_single', size=35)

##################################################

# consensus clustering

# later on, with parallel functionality, we'll end up with nParallel sets of microclusters from a single stream
# for now I just use the different streams to generate multiple sets as an array
n_parallel = 5
K = 4
n_cells = 301
lowrankrange = range(10,20)
clusterings = np.empty((n_cells, len(lowrankrange), n_parallel), dtype=int)

np.random.seed(322)
from sklearn.cluster import KMeans
for i in range(0, len(lowrankrange)):
    for j in range(0, n_parallel):
        microcentroids, assignments = sc3s.tl.strm_spectral(adata, k=100, initial = 100, 
                stream = 20, lowrankdim = lowrankrange[i])
        clusterings[:, i, j] = sc3s.tl.weighted_kmeans(microcentroids, assignments)

clusterings = clusterings.reshape((n_cells, len(lowrankrange)*n_parallel))

# consensus clustering of the cells
B = sc3s.tl.convert_clustering_to_binary(clusterings, K)
kmeans_macro = KMeans(n_clusters=K).fit(B)
kmeans_macro.labels_
adata.obs = adata.obs.assign(sc3_k4 = pd.Categorical(kmeans_macro.labels_))

sc.pl.pca(adata, color='sc3_k4', size=35)
sc.pl.pca(adata, color='label2', size=35)

##################################################

import matplotlib.pyplot as plt
with plt.style.context('fivethirtyeight'):
    plt.matshow(-B.T)
    plt.axis('off')
    plt.savefig("binary.png", format="png", transparent=True)

# single run
with plt.style.context('fivethirtyeight'):
    plt.matshow(-B[:,0:4].T)
    plt.axis('off')
    plt.savefig("binary_onerun.png", format="png", transparent=True)

