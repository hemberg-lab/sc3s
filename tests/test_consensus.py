import sc3s
import numpy as np
import anndata as ad
from sklearn.metrics import adjusted_rand_score

from pathlib import Path
TESTS_DIR = Path(__file__).resolve().parent
adata = ad.read_h5ad(TESTS_DIR.joinpath("data/data_pca.h5ad"))

def test_consensus():
    np.random.seed(644)

    sc3s.tl.consensus(
        adata, 
        n_clusters = [2, 4],
        d_range=[2, 5],
        n_runs = 5,
        n_facility = None,
        multiplier_facility = None,
        batch_size = 10,
        random_state = None 
    )
    
    assert 'sc3s_4' in adata.obs
    assert 'sc3s_2' in adata.obs
    
    ari_score = adjusted_rand_score(
        adata.obs['sc3s_4'], 
        adata.obs['label']
    )

    assert ari_score > 0.18