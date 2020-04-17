# this is the scipy version
A = generate_microclusters(adata, k=200, initial = 100, stream = 10, lowrankdim = 20)
macrocentroids, macroclusters = kmeans(A[0], 4)
macroclusters(A[1])

##########################

# testing how to randomise cells
import numpy as np
S = 5
numbers = np.arange(S) #np.array(list(map(lambda x: chr(ord('a') + x), range(0,S))), dtype='str')
lut = np.arange(S)
np.random.shuffle(lut)

X = np.empty(S)
(numbers[lut] ** 2)[lut]


arr
letters[arr]

# testing deepcopy
import copy
A = 322
b = A
A = 644

A = [1,2,3,4,5]
b = A
A[4] = 3000

# test with np arrays

print(A, id(A))
print(b, id(b))
