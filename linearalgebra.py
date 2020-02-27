#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 15:26:42 2020

@author: fq1

https://subscription.packtpub.com/video/data/9781839214219
"""


import numpy as np

# a 1x1 matrix is a scalar
# 0 dimensions
# geometrically, it's a point
s = 5
type(s)
np.array(5) # equivalent

# column vector is the notation
# 1 dimension
# represents a line
v = np.array([5,-2,4])
type(v)

# using two vectors as a matrix
# geometrically, represent a plane
m = np.array([[5,12,6],[-3,0,14]])
type(m)

m.shape # 2x3

v.shape # returns (3,)
v_col = v.reshape(1,3)
v_row = v.reshape(3,1)

v_col.shape
v_row.shape

# tensors are a generalisation
# scalar: tensor of rank 0,
# vector 1, matrix 2

# here is a tensor of rank 3
m1 = np.array([[5,12,6],[-3,0,2]])
m2 = np.array([[9,8,7],[-1,5,0]])
t = np.array([m1,m2])
t

# matrix addition and subtraction
m1 + m2
v - v
v_col - v_row # quite unexpected!

########################################

# TRANSPOSE

# this makes sense for arrays
# but not linear algebra!
1 + 1
m1 + np.array(1)

# transpose
m
np.transpose(m)

s
np.transpose(s) # pointless

v
np.transpose(v) # this is not meaningful
np.transpose(v_col) # becomes row vect

########################################

# MULTIPLICATION

# scalar multiplication
np.dot(3,5)

# vector multiplication: dot/inner product
v3 = np.array([1,2,3])
v4 = np.array([1,2,3])
np.dot(v3, v4) # don't need to transpose

v
np.dot(v,v)
np.dot(v_row, v_row) # doesn't work
np.dot(v_row, v_col) # 3x3
np.dot(v_col, v_row) # same as first, but is an array

# this doesn't change the shape
# it simply scales the vector
np.dot(5, v3)

########################################

# MATRIX MULTIPLICATION

# think columns of a2 as the scalars
a1 = np.array([[1,1],[2,2],[3,3]])
a2 = np.array([[1,3],[2,0]])
np.dot(a1,a2)

# i find this syntatically more intuitive 
# think rows of a1 as the scalars
# of the rows in a2
a3 = np.array([[1,3],[2,0]])
a4 = np.array([[1,1],[2,2]])
np.dot(a3,a4)

########################################

# VECTORISED OPERATIONS (array programming)
1 + 2 * np.array([1,2,3,4,5])

# let's say we calculate price of house
#   = 2(age) + 100(size)

# the age and sizes of 3 houses
h = np.array([[10, 500], [2, 999], [3, 699]])

# we can do
np.dot(h, np.array([2, 100]))
