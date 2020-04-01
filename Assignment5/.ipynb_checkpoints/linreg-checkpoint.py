#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Exercise 2
import numpy as np

def multivarlinreg(X, y):
    X = add_one(X)
    w = np.linalg.inv(X.T @ X) @ X.T @ y
    return w
    
def add_one(X):
    row = len(X)
    one_col = np.ones(row)
    X = np.c_[one_col, X]
    return X

def print_names(weights, names):
    for i in range(len(weights)):
        print(f"{names[i]}:  {weights[i]}")
    print("\n")
    
def normalize_data(source, target):
    '''Takes in a source array and a target array. 
    Normalizes the target array based on the source array'''
    mean = np.mean(source, axis = 0)
    sd = np.std(source, axis = 0)
    target = (target - mean)/sd
    return target

# In[10]:


# Exercise 3
def rmse(pred, true):
    error = (true - pred)**2
    rms = np.sqrt(np.mean(error))
    return rms

def test(weights, test_x, test_y):
    X = add_one(test_x)
    pred = X @ weights
    error = rmse(pred, test_y)
    return error




