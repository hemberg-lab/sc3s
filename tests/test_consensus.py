import sc3s
import pandas as pd
import numpy as np
import scanpy as sc
from sklearn.metrics import adjusted_rand_score

rootdir = "tests/data/"
adata = sc.read_h5ad(f"{rootdir}data.h5ad")

# preprocessing to remove uninformative genes, and log transform
sc.pp.filter_genes(adata, min_cells=50)
sc.pp.log1p(adata)
adata

def test_consensus():
    sc3s.tl.consensus(adata, num_clust = [4,11], stream = 200, lowrankrange=[20,35,50])
    assert 'sc3s_4' in adata.obs
    assert 'sc3s_11' in adata.obs
    assert adjusted_rand_score(adata.obs['sc3s_4'], adata.obs['label2']) > 0

def test_consensus_old():
    sc3s.tl.consensus_old(adata, num_clust = [4,11], lowrankrange=[20,35,50])
    assert 'sc3ori_4' in adata.obs
    assert 'sc3ori_11' in adata.obs
    assert adjusted_rand_score(adata.obs['sc3ori_4'], adata.obs['label2']) > 0


"""
sc3s.tl.consensus(adata, num_clust = [4,11], stream = 200, lowrankrange=[50,75,100])
sc3s.tl.consensus_old(adata, num_clust = [4,11], lowrankrange=[50,75,100])

sc.tl.pca(adata, n_comps=25, svd_solver='randomized')
sc.pl.pca(adata, color='sc3s_4', size=20)
sc.pl.pca(adata, color='sc3ori_4', size=20)
sc.pl.pca(adata, color='label2', size=20)
"""
