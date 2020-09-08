def strm_spectral_new(data,
                  k = 100, 
                  lowrankdim = 25, # rename as n_component
                  stream = 1000,
                  batch = 1000,
                  n_runs = 5,
                  svd = "sklearn",
                  randomcellorder = True):
    """
    Cluster the cells into microclusters. k should be set to a relatively large number, maybe a 
    fraction of the first batch?

    In the code, `centroids` obviously refers to microcentroids.

    To do: incorporate parallel functionality using the same Laplacian.

    Need to modularise steps more (see separate notes).
    """
    
    # initialise and print parameters
    i = 0
    j = stream
    n_cells, n_genes = data.shape
    print(f"""PARAMETERS:
    
    num_cells: {n_cells}, n_genes: {n_genes}
    
    stream size: {stream}
    number of components: {lowrankdim}
    SVD algorithm: {svd}

    """)

    # generate look up table for the random ordering of cells
    lut = np.arange(n_cells) 
    if randomcellorder is True:
        np.random.shuffle(lut)

    # create dictionary with centroids and assignments of the k-means runs
    runs = {t: {'cent': np.empty((k, lowrankdim)),
                'asgn': np.empty(n_cells, dtype='int32')} for t in range(0, n_runs)}

    # initialise structures to be updated during training
    sketch_matrix = np.empty((lowrankdim, n_genes)) # learned embedding
    gene_sum = np.empty(n_genes) # to calculate dataset centroid
    
    print("beginning the stream...\n")
    while i < n_cells:
        # obtain range of current stream
        if (n_cells - i) < stream: # resize last stream
            j = n_cells - i 
            sketch_matrix = sketch_matrix[:(lowrankdim + j), :] # i think not needed
    
            def resize_centroids(run, k, j):
                run['cent'] = run['cent'][:(k+j)]
                return run
            runs = {t: resize_centroids(run, k, j) for t, run in runs.items()}
        else:
            j = stream

        # obtain current stream
        if issparse(data[lut[i:(i+j)], ]):
            cells = data[lut[i:(i+j)], ].todense()
        else:
            cells = data[lut[i:(i+j)], ]
        
        # update embedding
        sketch_matrix, V, u, gene_sum = _embed(cells, C, gene_sum, lowrankdim, svd)

        # rotate centroids
        assert np.all(Vold != V), "rotation matrices are the same!"
        runs = {t: _rotate_centroids(run, V, Vold) for t, run in runs.items()}

        # assign new cells to centroids, centroids are reinitialised by random chance
        runs = {t: assign(run, u, k, lut, i, j) for t, run in runs.items()}

        print(f"working on observations {i} to {i+j}...")
        i += j
        Vold = V

    # unrandomise the ordering
    runs = {(lowrankdim, t): _reorder_cells(run, lut) for t, run in runs.items()}

    return runs

def _embed(cells, sketch_matrix, sum_gene, lowrankdim, svd = "sklearn"):
    assert len(sum_gene.shape) == 1
    assert cells.shape[0] == sum_gene.shape[0]
    assert sketch_matrix.shape[0] == lowrankdim

    stream_size = cells.shape[0]

    # calculate the sum_vector based on the new cells
    sum_gene = sum_gene + np.sum(cells, axis = 0)

    # calculate dataset centroid up to current point in stream
    dataset_centroid = sum_gene / np.sum(sum_gene)

    # approximate degree matrix (n_cell x n_cell)
    D = np.diag(np.dot(cells, dataset_centroid))
    assert np.shape == (stream_size, stream_size), "degree matrix incorrect size"

    # approximated normalised Laplacian
    laplacian = np.dot(np.sqrt(D), cells)

    # concatenate with previous sketch matrix and perform SVD
    C = np.empty(lowrankdim + stream_size, sketch_matrix.shape[1])
    C[:lowrankdim] = sketch_matrix
    C[lowrankdim:] = laplacian

    # choose svd algorithm (maybe use a switch case)
    # https://jaxenter.com/implement-switch-case-statement-python-138315.html
    if svd == "sklearn":
        U, s, V = svd_sklearn(C, lowrankdim)
    elif svd == "scipy":
        U, s, V = svd_scipy(C, lowrankdim)
    
    # update sketch matrix
    s2 = s**2
    s_norm = np.sqrt(s2 - s2[-1])
    sketch_matrix = np.dot(np.diag(s_norm), V)

    # normalise the landmarks in each cell (row-wise)
    u = U / np.transpose(np.tile(np.linalg.norm(U, axis=1), (lowrankdim, 1)))

    return sketch_matrix, V, u. sum_genes










    # get the centroid and calculate the degree matrix and normalised laplacian

    # make a new matrix, and then append the previous C, and new cells and into it

    # run SVD

    # calculate coordinates of new cells, it's the 
    
    # return sketch_matrix, V, U
    # (return the whole U as it is, which is a n_cell x k column, subset as necessary)
    
    C = np.empty()
    sketch_matrix = 

def _assign(run, cell_projections, k, lut, i, j):
    # incorporate something related to miniKmeansBatch

