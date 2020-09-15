import numpy as np
import pandas as pd

def _parse_int_list(x, error_msg = "value must be integer > 1, or a non-empty list/range of such!"):
    """
    Parse
    """
    if x is None:
        x = [3, 5, 10]
    elif isinstance(x, int) and x > 1:
        x = [x]
    elif isinstance(x, range):
        if len(x) > 0:
            pass
        else:
            raise Exception(error_msg)
    elif isinstance(x, list):
        for num in x:
            if isinstance(num, int) and num > 1:
                pass
            else:
                raise Exception(error_msg)
    else:
        raise Exception(error_msg)
    
    return x

def _generate_num_range(num_clust, num_per_side=1, step=3, prefix = None):
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


def _write_results_to_anndata(result, adata, num_clust='result', prefix='sc3s_'):
    """
    Mutating function. Add clustering labels as a new column in `adata`'s obs dataframe.

    Unit test whether this adds propwerly, if there is already something with the same name, or not.
    """
    adata.obs = adata.obs.drop(prefix + str(num_clust), axis=1, errors='ignore')
    adata.obs.insert(len(adata.obs.columns), prefix + str(num_clust), result, allow_duplicates=False)

