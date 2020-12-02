# special objects for the trials dictionary 
from collections import namedtuple

# key
rk = namedtuple('trial_params', 'd, i')

# value
rv = namedtuple('trial_params_value', 'facility, labels, inertia')


def _format_integer_list_to_iterable(x, var_name):
    """
    Format a list/range/single integer into iterable format. Also checks that values are >1.
    """
    if isinstance(x, int) and x > 1:
        x = [x]
    elif isinstance(x, range):
        if len(x) > 0:
            pass
        else:
            raise Exception(f"{var_name}: specified range has length 0.")
    elif isinstance(x, list):
        for num in x:
            if isinstance(num, int) and num > 1:
                pass
            else:
                raise Exception(f"{var_name}: values must be greater than 1.")
    else:
        raise Exception(f"{var_name}: must be integer > 1, or a non-empty list/range.")
    
    return x

def _write_results_to_anndata(result, adata, n_clusters, prefix='sc3s'):
    """
    Mutating function. Add clustering labels as a new column in `adata`'s obs dataframe.

    Unit test whether this adds propwerly, if there is already something with the same name, or not.
    """

    import pandas

    assert isinstance(result, (np.ndarray, list, pd.Categorical))
    assert len(result) == adata.obs.shape[0]
    assert isinstance(n_clusters, int)

    adata.obs[f'{prefix}_{str(n_clusters)}'] = cell_labels
    adata.obs[f'{prefix}_{str(n_clusters)}'] = adata.obs[f'{prefix}_{str(n_clusters)}'].astype('category')

    #adata.obs = adata.obs.drop(f'{prefix}_{str(n_clusters)}', axis=1, errors='ignore')
    #adata.obs.insert(len(adata.obs.columns), f'{prefix}_{str(n_clusters)}', result, allow_duplicates=False)

