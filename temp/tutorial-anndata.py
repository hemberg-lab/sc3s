# http://falexwolf.de/blog/171223_AnnData_indexing_views_HDF5-backing/
import numpy as np
import pandas as pd
import anndata as ad

##########

# generate sample data

n_obs = 1000

# create annotation for time of sample collection
obs = pd.DataFrame()
obs['time'] = np.random.choice(['day 1', 'day 2', 'day 4', 'day 8'], n_obs)

# create fake gene names
from string import ascii_uppercase
var_names = [i*letter for i in range(1,10) for letter in ascii_uppercase]

n_vars = len(var_names)
var = pd.DataFrame(index=var_names)
X = np.arange(n_obs*n_vars).reshape(n_obs, n_vars)

# create the AnnData object
adata = ad.AnnData(X=X, obs=obs, var=var, dtype='int32')

##########

print(adata) # verbose with dims

print(adata.X)# X is known as a layer

# observation names are just numbers from 0 to 999
print(adata.obs_names[:10].tolist())
print(adata.obs_names[-10:].tolist())

# variable names are the gene names we created
print(adata.var_names[:10].tolist())
print(adata.var_names[-10:].tolist())

##########

# indexing adata objects always returns views
# this is memory efficient
adata

# a view of a vector (including the metadata)
adata[:, 'A']

# index a single vector (matrix values alone)
# for now, this returns a 1d array
# but in v0.7, it returns 2D arrays (maintains the dims)
adata[:3, 'A'].X

# new method, seeems to return an actual array (not just a view)
adata.obs_vector('A')[:3] # returns the column 'A'
adata.var_vector(1)[:5] # for rows (genes)

# set values
adata[:3, 'A'].X = [0,0,0]
print(adata[:5, 'A'].X)

# trying to set a view of an AnnData as a VARIABLE creates a new object
# slightly confused about this, does it only happen for var and obs?
# no, it happpens with .X too
adata_subset = adata[:5, ['A','B']]
adata_subset # still a view
adata_subset.obs['foo'] = range(5)
adata_subset # no longer a view

# can also slice with sequences or boolean indices
adata[adata.obs['time'].isin(['day 1', 'day 2'])].obs.head()

##########

# write the results to disk
adata.write('./write/my_results.h5ad')
!h5ls './write/my_results.h5ad'

adata.write_csvs('./write/my_results_csvs',)
!ls './write/my_results_csvs'

# convenience method for computing the size of objects
def print_size_in_MB(x):
    print('{:.3} MB'.format(x.__sizeof__()/1e6))
print_size_in_MB(adata) # about 1 MB

###########

# viewing without loading into memory is possible with AnnData
# this can also be done when backing the object with a file
adata.isbacked

# switch to "backed" mode
# by setting a backing file name
adata.filename = './write/test.h5ad'
adata.isbacked

print_size_in_MB(adata) # only 0.128MB

adata # explicitly mentions where it's backed

# indexing in "backed" mode
print(adata[0, 'A':'C'].X)

# changes the value of the X matrix
# doing this CHANGES the timestamp of the HDF5 file
adata[0, 'A':'C'].X = [3,2,2]

# NOTE: backing only affects the data matrix X
# all annotations (obs, var) are kept in memory
# changes to them will have to be written to disk using `.write()`

adata.var # currently empty
type(adata.var) # it's a dataframe

# let's set the first column
# doing this does not change the timestamp of the HDF5 file
adata.var.loc[:, 'type'] = np.random.choice(['housekeeping', 'variable'], n_vars)

adata.file.close() # close the file
!h5ls './write/test.h5ad' # this only works when it's closed
adata.file.isopen

# accessing anything reopens the file
adata.X
adata.file.isopen
adata.var

##########

# copy the backed object
adata_new = adata.copy(filename='./write/test1.h5ad')

# you can move the backing file simply by resetting the filename
adata.filename = './write/test1.h5ad'

# compared to the non-backed case
adata_subset = adata[:5, ['A','B']]
adata_subset 
adata_subset.obs['foo'] = range(5) # does not work

# have to explicitly copy
adata_subset = adata[:5, ['A', 'B']].copy(filename='./write/test1_subset.h5ad')
adata_subset.obs['foo'] = range(5) # works
adata_subset
adata_subset.write()

# TRY REOPENING IT