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

# perform PCA
sc.tl.pca(adata, n_comps=25, zero_center=None, svd_solver='arpack', 
          random_state=None, chunked=False, chunk_size=None)

def test_consensus():
    sc3s.tl.consensus(
        adata, 
        n_clusters = [4, 11],
        n_facility = None,
        d_range=[20, 25],
        n_runs = 5
    )
    
    assert 'sc3s_4' in adata.obs
    assert 'sc3s_11' in adata.obs
    
    assert adjusted_rand_score(adata.obs['sc3s_4'], adata.obs['label2']) > 0