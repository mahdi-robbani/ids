import numpy as np
import matplotlib.pyplot as plt

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
    eigenvectors = eigenvectors[:, :d] # select first d eigenvectors
    target = eigenvectors.T @ target.T # transpose EV to d*D matrix, transpose data to D*N matrix, get d*N matrix
    target = target.T #transpose to N*d matrix
    return target

def mds_plot(data, save, d = 2):
    data = mds(data, data, d)
    plt.scatter(data[:,0], data[:,1])
    plt.axis('equal')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('Plot of toy dataset projected on first two principal components')
    if save == False:
        plt.show()
    else:
        plt.savefig("toy"+ str(len(data))+".png", dpi = 200)
