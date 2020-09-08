import numpy as np
import pandas as pd

def calculate_rmse(A, B):
    """
    Calculate root mean squared error between two matrices.
    """
    error = A - B
    return np.sum(error ** 2)


def _check_iterable(obj):
    """
    Check whether something is an iterable (e.g. list).
    """
    try:
        iter(obj)
    except Exception:
        return False
    else:
        return True

def _write_results_to_anndata(result, adata, num_clust='result', prefix='sc3s_'):
    """
    Mutating function. Add clustering labels as a new column in `adata`'s obs dataframe.

    Unit test whether this adds propwerly, if there is already something with the same name, or not.
    """
    adata.obs = adata.obs.drop(prefix + str(num_clust), axis=1, errors='ignore')
    adata.obs.insert(len(adata.obs.columns), prefix + str(num_clust), result, allow_duplicates=False)

def get_num_range(num_clust, num_per_side=1, step=2, prefix = None):
    """
    What does this do again? Unit test this please.
    """
    start = num_clust - num_per_side * step
    end = num_clust + num_per_side * step
    clust_range = list(filter(lambda x: x>1, range(start, end + 1, step)))
    if prefix is None:
        return clust_range
    else:
        return list(map(lambda x: prefix + str(x), clust_range))
