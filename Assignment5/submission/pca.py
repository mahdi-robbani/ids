import numpy as np
import matplotlib.pyplot as plt
from plot import *

def pca(data):
    '''Return an array of eigen vectors and eigenvalues for a long dataset'''
    cov_matrix = np.cov(data.T) # transpose to get the correct covariance matrix
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix) # calculate eigen stuff
    eigenvalues = eigenvalues[::-1] # reverse sort array
    eigenvectors = eigenvectors[:,::-1] # reverse sort matrix
    return eigenvectors, eigenvalues

def standardize_data(data):
    '''Transforms the data so that mean = 0 and sd = 1'''
    data = center_data(data)
    data = normalize_data(data)
    return data
	
def center_data(data):
    mean = np.mean(data, axis = 0)
    data = data - mean
    return data

def normalize_data(data):
    sd = np.std(data, axis = 0)
    data = data/sd
    return data

def mds(target, source, d):
    """
    Transform a target using the eigenvectors of the source
    """
    eigenvectors, eigenvalues = pca(source) # calculate eigenvectors
    e_d = eigenvectors[:, :d] # select first d eigenvectors
    target = e_d.T @ target.T # transpose EV to d*D matrix, transpose data to D*N matrix, get d*N matrix
    target = target.T #transpose to N*d matrix
    return target, eigenvectors

def mds_inv(reduced_data, eigenvector, d):
    """Transfoms a reduced data set back to the original size using the original eigenvectors"""
    eigen_inv = np.linalg.inv(eigenvector) #calculat inverse
    e_inv = eigenvector[:, :d] #get desired number of dimensions
    data = e_inv @ reduced_data.T # D*d @ d*N
    data = data.T # convert to N*D
    return data

def mds_plot(data, title, save, d = 2):
    data = mds(data, data, d)
    plt.scatter(data[:,0], data[:,1], zorder = 2, ec = 'black')
    plot_template(title = title,
                 xlabel = "Principal Component 1",
                 ylabel = "Principal Component 2",
                 equal_axis = True, 
                 legend = False,
                 save = save)
