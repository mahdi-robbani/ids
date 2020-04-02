#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import matplotlib.pyplot as plt
from kmc import *
from pca import *
from plot import *
s = False


# In[ ]:


def get_proportion(n, cluster):
    """
    Takes a number (n) and a list of values (cluster)
    and returns the proportion of values that in the lsit that match n
    """
    N = len(cluster)
    proportion = len(cluster[cluster == n])/N
    print(f"Proportion of {n}: {round(proportion * 100, 2)}%")
    return proportion

def get_proportions(numbers, cluster):
    """
    Returns all proportions for a given cluster
    """
    proportions = []
    for i in numbers:
        p = get_proportion(i, cluster)
        proportions.append(p)
#     print(f"Sum: {sum(proportions)}")
    return proportions

def get_cluster_proportions(numbers, clusters, prep_data):
    """
    Takes in a list of numbers and clusters.
    Returns the proportion of each number for each cluster
    """
    cluster_proportions = []
    for i in range(len(clusters)):
        print(f"Cluster: {i+1}")
        cluster = prep_data[clusters[i]][:, 0]
        cluster_proportion = get_proportions(numbers, cluster)
        cluster_proportions.append(cluster_proportion)
    return cluster_proportions

def plot_number(means, title):
    for i in range(len(means)):
        plt.imshow(np.reshape(means[i], (28,28)))
        plot_template(title=title + str(i+1), xlabel="", ylabel="", grid = False, equal_axis=False, save = s)

def get_prop_plot(k, numbers, data, labels, title):
    means = kmean(k, data)
    clusters = get_cluster(means, data)
    prep_data = np.c_[labels, data]
    get_cluster_proportions(numbers, clusters, prep_data)
    plot_number(means, title)


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

def get_acc(k, train_x, train_y, test_x, test_y):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(test_x, test_y)
    acc = accuracy_score(train_y, model.predict(train_x))
    return acc

def cross_validate(k, XTrain, YTrain):
    '''For a value k, perform 5 fold cross validation for k nearest neighbors'''
    loss_list = []
    # create indices for CV
    cv = KFold(n_splits = 5)
    # loop over CV folds
    for train, test in cv.split(XTrain):
        XTrainCV, XTestCV, YTrainCV, YTestCV = XTrain[train], XTrain[test], YTrain[train], YTrain[test]
        lossTest = 1 - get_acc(k, XTrainCV, YTrainCV, XTestCV, YTestCV)
        loss_list.append(lossTest)
    average_loss = np.mean(loss_list)
    return average_loss

def find_best_k(k_list, XTrain, YTrain):
    '''Given a list of ks, perform 5 fold cross validation for each k and return the best k'''
    k_loss = []
    for k in k_list:
        loss = cross_validate(k, XTrain, YTrain)
        k_loss.append(loss)
        print("Loss for "+ str(k) + " neighbors: " + str(loss))
    ind = k_loss.index(min(k_loss))
    best_k = k_list[ind]
    acc = 1 - min(k_loss)
    print("====== Results ======")
    print(f"Best k: {best_k}\nAccuracy: {acc}")
    return best_k


# In[ ]:


def get_prop_plot_mds(k, numbers, data, labels, dimension, title):
    print("===============")
    print(f"{dimension} dimensions:")
    print("===============")
    reduced_data, eigenvectors = mds(data, data, dimension)
    reduced_means = kmean(k, reduced_data)
    clusters = get_cluster(reduced_means, reduced_data)
    prep_data = np.c_[labels, reduced_data]
    get_cluster_proportions(numbers, clusters, prep_data)
    reduced_means = np.array(reduced_means)
    means = mds_inv(reduced_means, eigenvectors, dimension)
    plot_number(means, title=str(dimension) + title)


# In[ ]:


def mds_acc(dim, k_list, data, labels):
    print("===============")
    print(f"{dim} dimensions:")
    print("===============")
    reduced_data, _ = mds(data, data, dim)
    #print(reduced_data)
    best_k = find_best_k(k_list, reduced_data, labels)
    best_acc = get_acc(best_k, reduced_data, labels, reduced_data, labels)
    print(f"Best K: {best_k} Accuracy: {best_acc}")

