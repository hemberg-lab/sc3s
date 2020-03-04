#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 16:09:47 2020

@author: fq1

https://pandas.pydata.org/pandas-docs/stable/getting_started/10min.html
"""


import numpy as np
import pandas as pd

##################################################

# OBJECT CREATION

# create a Series
s = pd.Series([1,3,5, np.nan, 6, 8])
s

# create a DataFrame using a numpy array
dates = pd.date_range('20130101', periods=6)
df = pd.DataFrame(np.random.randn(6,4), index=dates, columns=list('ABCD'))
df

# create DataFrame by passing a dict of objects
# each column will have different dtypes
df2 = pd.DataFrame({'A': 1.,
                    'B': pd.Timestamp('20130102'),
                    'C': pd.Series(1, index=list(range(4)), dtype='float32'),
                    'D': np.array([3] * 4, dtype='int32'),
                    'E': pd.Categorical(["test", "train", "test", "train"]),
                    'F': 'foo'})

##################################################

# VIEWING DATA

df.head()
df.tail(3)

# can convert DataFrame to NumPy
# can be expensive: NumPy arrays can have only one dtype
# note: indexes and column labels are dropped
df.to_numpy()

# quick statistic summary
df.describe()

# transpose data
df.T

# sorting by an axis
df.sort_index(axis=0, ascending=False)
df.sort_index(axis=1, ascending=False)

# sort by values of a column (doesn't work for rows)
df.sort_values(by='B')

##################################################

# SELECTION
# the pandas data access methods are optimized: .at, .iat, .loc, iloc

# select a single column yields a Series
df['A']
df.A # equivalent, only works for columns

# slices the rows in the Pythonic/NumPy way
df[0:3]
df['20130102':'20130104']
type(df[0:3]) # still a DataFrame

##########

# select by label/row to get a series
df.loc[dates[0]]

# select multi-axis by label
df.loc[:, ['A','B']]
df.loc['20130102':'20130104', ['A', 'B']]

# to get a single scalar value
df.loc[dates[0], 'A']
df.at[dates[0], 'A']  # this is faster

##########

# select by position of integer
df.iloc[1] # 2nd row
df.iloc[1, :] # equivalent, doesnt stay as DataFrame

df.iloc[:, 1] # 2nd column, becomes Series

df.iloc[3:5, 0:2]
df.iloc[[1,2,4], [0,2]]

df.iloc[1:3, :]
df.iloc[:, 1:3]

# get a value explicitly
df.iat[1,1]

##########

# Boolean indexing
df[df['A'] > 0]
df[df > 0] # false becomes NaN

# using the isin() method for filtering
df2 = df.copy()
df2['E'] = ['one', 'one', 'two', 'three', 'four', 'three']
df2[df2['E'].isin(['two','four'])]

##########

# setting
s1 = pd.Series([1,2,3,4,5,6], index=pd.date_range('20130102', periods=6))
df['F'] = s1
df.at[dates[0]]