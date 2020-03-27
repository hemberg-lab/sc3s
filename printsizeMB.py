# function to print size of a Python object in memory
def print_size_in_MB(x):
    print('{:.3} MB'.format(x.__sizeof__()/1e6))