#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Exercise 7
import numpy as np
import matplotlib.pyplot as plt
from plot import *
s = False


# In[ ]:


#7.1
def plot_iris(data, name):
    if "2D1" in name:
        y = "Feature 2"
    else:
        y = "Feature 1"
    
    plt.scatter(data[data[:,2] == 0][:, 0], data[data[:,2] == 0][:, 1], ec = "black", label = "Class 0", zorder = 3)
    plt.scatter(data[data[:,2] == 1][:, 0], data[data[:,2] == 1][:, 1], ec = "black", color = 'red', label = "Class 1", zorder = 3)
    plot_template(title = "Plot of " + name, xlabel = "Feature 0", ylabel = y, equal_axis=False, legend= True, save = s)


# In[ ]:


#7.2
def logistic(x):
    out = 1/(1 + np.exp(-x))
    return out

def add_one(X):
    row, col = np.shape(X)
    one_col = np.ones(row)
    X = np.c_[one_col, X]
    return X

def gradient(X, y, w):
    """
    Returns a vector of partial derivaties
    """
    s = -y.T * (w.T @ X.T) # Transpose to get 1*d @ d*N
    theta = logistic(s) # 1*N
    c =  -y * X # N*d
    grad = c.T @ theta.T # Transpose to d*N @ N*1
    return grad

def insample_error(X, y, w):
    """
    Returns a single real value which corresponds to the error
    """
    N = len(X)
    s = -y.T * (w.T @ X.T) # Transpose to get 1*N x 1*d @ d*N
    pyx = np.log(1 + np.exp(s)) # Calculate P(Yn|Xn) likelihood
    error = np.sum(pyx)/N # Calculate sum[P(Yn|Xn)]/N
    return error

def train_log(X, y):
    """
    Perfoms logistic regression training
    Takes in X = N*d array and Y = N*1 array
    Returns an array of weights w = d*1
    """
    X = add_one(X) # Add intercept column
    N, d = np.shape(X)
    w = np.reshape(np.random.randn(d), (d, 1)) #initialize random weights
    error = insample_error(X, y , w)
    learning_rate = 0.01
    iteration = 1
    convergance = 0
    tolerance = 10**-10
    
    while convergance == 0:
        m = gradient(X, y, w)
        w_new = w - (learning_rate * m) # update weight
        new_error = insample_error(X, y, w_new)
        g = np.linalg.norm(m) # convert partial derivate array to single gradient value
        iteration += 1
        
        #check if new error is better
        if new_error < error:
            w = w_new
            error = new_error
            learning_rate *= 1.1
        else:
            learning_rate *=0.9
        
        #check convergance condition
        if g < tolerance:
            #print("Tolerance reached")
            convergance = 1
        elif iteration == 10000:
            #print("Max iterations")
            convergance = 1
    return w
    
def predict_log(X, w):
    X = add_one(X) #Add column for intercept
    pred = logistic(w.T @ X.T).T # h(x) = theta(w.T @ x)
    pred = pred > 0.5 # Convert prediction to a boolean that indicates if it is > 0.5 or not
    pred = np.array(pred, dtype = int) # convert to 0 or 1
    pred = 2*(pred-0.5) # conver to -1 or 1
    return pred

def get_error(true, pred):
    """
    Takes in two N*1 arrays
    Each array consists of -1 or 1
    Returns a single error value
    """
    N = len(true)
    error = abs(true - pred)/2 #convert each value to 0 or 1
    error = np.sum(error)/N
    return error

def split_data(data):
    """
    Saves last column as y. Converts y to -1 or 1
    Saves rest of the columns as x
    """
    x = data[:, :2]
    y = data[:, -1:]
    y = 2*(y-0.5) # convert to -1 or 1
    return x, y

def log_regression(train, test):
    """
    Perform logistic regression on a dataset with y value as last column
    Returns the 0-1 error value, weights, and predicted values
    """
    train_x, train_y = split_data(train)
    test_x, test_y = split_data(test)
    weights = train_log(train_x, train_y)
    pred = predict_log(test_x, weights)
    error = get_error(test_y, pred)
    return error, weights, pred

