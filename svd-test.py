# https://stackoverflow.com/questions/18706863/difference-between-truncatedsvd-and-svds
import numpy as np
import scipy as sc
from scipy import linalg
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import random as sparse_random
import matplotlib.pyplot as plt

"""
# scipy tutorial
m, n = 9, 6
a = np.random.randn(m, n)
U, s, Vh = linalg.svd(a)
U.shape, s.shape, Vh.shape

# reconstruct the original matrix
sigma = np.zeros((m,n))
for i in range(min(m, n)): # need to do this because not a n x n matrix
    sigma[i,i] = s[i]
a1 = np.dot(U, np.dot(sigma, Vh))
np.allclose(a, a1)

# full matrices
U, s, Vh = linalg.svd(a, full_matrices=False)
U.shape, s.shape, Vh.shape
S = np.diag(s)
np.allclose(a, np.dot(U, np.dot(S, Vh)))
"""


# let me try the scikit version
#X = [[1,2,3],[1,4,2],[4,1,7],[5,6,8]]
m, n = 1000, 1000
X = np.random.randn(m, n)
k = np.shape(X)[1]//10 * 9 # number of singular vectors to keep

# TRUNCATED SVD, using half of the components
from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(n_components=k, n_iter=7, random_state=42)

US=svd.fit_transform(X)
V=svd.components_
S=svd.singular_values_ 
print('u,s,v', US,S,V)
#print('X_restored dot way\n', np.round(np.dot(US,V),1))
print('inverse of scikit-learn SVD\n',np.round(svd.inverse_transform(US),1))


# LINALG SVD
U1,S1,V1=np.linalg.svd(X)
#print('u1,s1,v1 remark negative mirrored\n',
#      U1[:,:2]*S1[:2],V1[:2,:])
print('X restored u1,s1,v1, using n/2 components\n',
      np.round( np.dot( U1[:,:k]*S1[:k],V1[:k,:] ),1 ) ) 

print('original\n',np.round(X,1)) #original

# sparse svd
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import svds, eigs

A = csc_matrix(X, dtype=float) # convert to sparse matrices
u2, s2, vt2 = svds(A, k=2)

print('sparse reverses !',u2*s2,vt2)
print('x restored',np.round( np.dot(u2*s2,vt2),1) )


