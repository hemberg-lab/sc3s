import anndata as ad
import numpy as np
import pandas as pd
from sc3s._misc import _write_results_to_anndata

n_obs = 1000

# generate new categorial columns to be added
col1 = pd.Categorical(np.random.choice([0,1,2,3,4,5], n_obs))
col2 = pd.Categorical(np.random.choice([0,1,2,3,4,5], n_obs))

# generate AnnData objects
X = np.arange(n_obs*500).reshape(n_obs, 500)
adata_nocol = ad.AnnData(X=X, dtype='int32')
adata_col1 = ad.AnnData(X=X, obs=pd.DataFrame(col1, columns=["sc3s_6"]), dtype='int32')
adata_col2 = ad.AnnData(X=X, obs=pd.DataFrame(col2, columns=["sc3s_6"]), dtype='int32')


def test_write_anndata_if_column_not_exist():
    _write_results_to_anndata(col2, adata_nocol, num_clust = 6)
    assert adata_nocol.obs.equals(adata_col2.obs)
    assert adata_nocol.obs.shape[1] == 1

def test_write_anndata_overwrite_existing():
    _write_results_to_anndata(col2, adata_col1, num_clust = 6)
    assert adata_col1.obs.equals(adata_col2.obs)
    assert adata_col1.obs.shape[1] == 1
