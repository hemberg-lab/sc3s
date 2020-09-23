import scanpy as sc
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

def _spectral_old(data,
                  k = 4,
                  d_range = 25,
                  n_runs = 5,
                  svd = "sklearn",
                  return_centers = False):

    assert isinstance(k, int)
    assert isinstance(n_runs, int)
    assert isinstance(return_centers, bool)
    d_range = _parse_int_list(d_range)

    # construct kernel affinity matrix
    W = cosine_similarity(data)

    # calculate normalised Laplacian
    D = np.diag(W.sum(axis=1)) # degree matrix
    I = np.identity(np.shape(W)[0])
    D_sqrt = np.diag(np.diag(D)**-0.5)
    L = I - np.dot(D_sqrt, np.dot(W, D_sqrt))

    # perform eigendecomposition
    vals, vecs = np.linalg.eig(W)

    # k-means across different values of d
    runs_dict = {}
    for d in d_range:
        cell_projections = vecs[:, :d]
        for t in range(n_runs):
            kmeans = KMeans(n_clusters=4).fit(cell_projections)
            runs_dict[(t,d)] = {'asgn': kmeans.labels_}
            if return_centers is True:
                runs_dict[(t,d)]['cent'] = kmeans.cluster_centers_

    return runs_dict