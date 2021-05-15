## sc3s - efficient scaling of single cell consensus clustering to millions of cells
SC3s is a package for the unsupervised clustering of single cell datasets. It is an updated version of [SC3](https://github.com/hemberg-lab/SC3), now reimplemented in Python and integrated into Scanpy. The algorithm has also been reengineered to allow it to scale efficiently towards millions of cells.


## Installation
Best way to install is to use `pip`. In a shell environment, run:
```sh
pip install sc3s
```

## Usage
SC3s is meant to be complemented with the popular Python single cell toolkit [Scanpy](https://scanpy.readthedocs.io/en/stable/), making use of the `AnnData`
data structure.

SC3s is designed to be an intermediate step in the single cell workflow. A general workflow would involve importing the data, running some preprocessing and doing dimensionality reduction with PCA. We recommending going through this offical [tutorial](https://scanpy-tutorials.readthedocs.io/en/latest/pbmc3k.html) on Scanpy's documentation, but a minimal barebones workflow could be:
```py
import scanpy as sc
adata = sc.read_h5ad("/path/to/data.h5ad")

# remove lowly expressed genes
sc.pp.filter_genes(adata, min_cells=10)

# log transform the data
sc.pp.log1p(adata)

# dimensionality reduction with PCA
sc.tl.pca(adata, svd_solver='arpack')
```

Once that is done, you can run SC3s with:
```py
import sc3s
sc3s.tl.consensus(adata, n_clusters=4)
```

SC3s can quickly evaluate multiple values for `n_clusters`, useful for exploring structure in data.
```py
sc3s.tl.consensus(adata, n_clusters=[4,8,10])
```

## Output
Running SC3s will modify the `obs` dataframe of the AnnData object, from which you can assess the assigned cluster labels:
```py
adata.obs             # shows the dataframe
adata.obs['sc3s_4']   # returns the cluster labels
```

The labels can then be used by subsequent functions in Scanpy, for example, plotted on UMAP and/or PCA plots.
```py
sc.pl.pca(adata, color='sc3s_4')
```

## Parameters
The default settings should suffice for most use cases, but you can get more information about the parameters with:
```py
help(sc3s.tl.consensus)
```

## Citation
Coming soon.
