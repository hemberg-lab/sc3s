# special objects for the trials dictionary 
from collections import namedtuple

# key
rk = namedtuple('trial_params', 'd, i')

# value
def rv(**kwargs):
    """
    Basically a nice wrapper around kwargs to avoid having to type out dictionary syntax.

    Returns dictionary object.
    """
    return kwargs

def _check_and_format_integer_list(num_list, min_val=None, max_val=None, var_name="variable", return_as_list = True):
    """
    Format a list/range/single integer into iterable format.
    
    Also checks that values fall within the limits specified.
    """
    # formats into an iterable
    _check_integer_boolean = isinstance(num_list, int)

    if _check_integer_boolean:
        num_list = [num_list]
    elif isinstance(num_list, range):
        if len(num_list) == 0:
            raise Exception(f"{var_name} specified range has length 0.")
        else:
            pass
    elif isinstance(num_list, list):
        if len(num_list) == 0:
            raise Exception(f"{var_name} specified list has length 0.")
        else:
            for num in num_list:
                if not isinstance(num, int):
                    raise Exception(f"{var_name} values must be integers.")
                else:
                    pass
    else:
        raise Exception(f"{var_name} must be integer, or a non-empty list/range.")

    # check that each value in the iterable satisfies the limits
    for num in num_list:
        if min_val is not None:
            if num < min_val:
                raise Exception(f"{var_name} must be greater than {min_val}.")
        if max_val is not None:
            if num > max_val:
                raise Exception(f"{var_name} must be less than {max_val}.")

    # the formated integer can be returned in the original format if specified
    if _check_integer_boolean and return_as_list == False:
        num_list = num_list[0]
    
    return num_list


def _check_integer_single(num, min_val=None, max_val=None, var_name="variable"):
    """
    Checks that a value is an integer.
    
    Also checks that the value fall within the limits specified.
    """

    # check that it is an integer
    if not isinstance(num, int):
        raise Exception(f"{var_name}: must be an integer.")

    # check that the integer value satisfies the limits
    if min_val is not None:
        if num < min_val:
            raise Exception(f"{var_name} must be greater than {min_val}.")
    if max_val is not None:
        if num > max_val:
            raise Exception(f"{var_name} must be less than {max_val}.")

    return num

def _check_dict_object(dict_object, true_n_clusters, true_n_cells):
    """
    Check that the runs in the dict_object contain the specified number of n_clusters and n_cells.
    """

    # check n_clusters and n_cells are integers
    true_n_clusters = _check_integer_single(true_n_clusters)
    true_n_cells    = _check_integer_single(true_n_cells)

    # check the number of cells
    n_cells_list = [len(x['labels']) for x in dict_object.values()]
    assert n_cells_list.count(true_n_cells) == len(n_cells_list), "number of cells is not consistent between trials"
    n_cells = true_n_cells

    # check the number of clusters
    n_clusters_list = [x['facility'].shape[0] for x in dict_object.values()]
    assert n_clusters_list.count(true_n_clusters) == len(n_clusters_list), "number of clusters not consistent between trials"
    n_clusters =  true_n_clusters

    return dict_object


    # check the number of cells
    # n_cells_list = [len(x['labels']) for x in dict_object.values()]
    # assert n_cells_list.count(true_n_cells) == len(n_cells_list), "number of cells not consistent"
    # n_cells = true_n_cells

    # # check the number of clusters
    # # there is an edge case for this, which is if the kth cluster is not assigned any cells
    # # will fix if error occurs, otherwise can just count that at least 75% of runs is correct?
    # n_clusters_list = [np.max(x['labels']) + 1 for x in dict_object.values()]
    # assert n_clusters_list.count(true_n_clusters) == len(n_clusters_list), "number of cells not consistent"
    # n_clusters =  true_n_clusters
