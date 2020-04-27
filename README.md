# sc3s
**Single Cell Consensus Clustering, Speeded up by Sequential Streaming**

## Arguments (in decreasing priority)
### Compulsory arguments
`data`: AnnData format.

`n` or `numclusters`: number of (macro)clusters to finf

### Optional
`b` or `batchmode`: use the whole stream as a single batch. Leave this as `False` for now.

`i` or `initialbatch`: defaults to 20-30 percent of cells. Round to nearest 10th. Max of 10000?

`s` or `streamsize`: defaults to 1 to 2 percent of cells. Rounds to nearest 10th. Max of 100?

`l` or `lowrankdim`: defaults to 5 percent of cells (this is one fourth of 20% or cells). Alternatively, can be a function of `initial_batch`?

`--iteration`: number of iterations for consensus. Defaults to 5.

`--micro`: number of microclusters to maintian. Defaults to `round(50 * log(cells))` for now. This might not even be needed, as there is a k-means [function](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html) in scikit. Of interest is their `n_init` argument, which runs it with different centroid seeds, and choose the best in terms of inertia (smaller values means denser clusters).

`--fjlt`: leave this for now. Might not even be in the main function.

### Unneeded arguments from original Julia prototype
`geneFilter`: [scanpy version](https://icb-scanpy.readthedocs-hosted.com/en/stable/api/scanpy.pp.filter_genes.html#scanpy.pp.filter_genes)

`logtransform`: [scanpy version](https://icb-scanpy.readthedocs-hosted.com/en/stable/api/scanpy.pp.log1p.html#scanpy.pp.log1p)


## Benchmarking and tutorials (in decreasing priority)
**Scanpy clustering tutorial**. Need to see how they write custers in the AnnData output.

How does scanpy package arguments? Do they use ArgParse??

**Loading data**. For benchmarking, best to prepare AnnData files ready with the matrix and clusters all in one. Need to write custom scripts to convert to this nice format.

**SVD algorithms**: to compare against matrix size and proportion of eigenvectors. And posssibly sparsity.

**AnnData**: get the backed mode working, after a prototype package is created.
