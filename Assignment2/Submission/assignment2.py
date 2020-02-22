#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import numpy.linalg as npl
# read in the data
dataTrain = np.loadtxt('IDSWeedCropTrain.csv', delimiter = ',')
dataTest = np.loadtxt('IDSWeedCropTest.csv', delimiter = ',')
# split input variables and labels
XTrain = dataTrain[:,:-1]
YTrain = dataTrain[:,-1]
XTest = dataTest[:,:-1]
YTest = dataTest[:,-1]


# In[2]:


#Exercise 1
def predict_one(Xvector, k, XTrain, YTrain):
    '''Takes in a single vector of 13 values, and makes a prediction using X and Y training data for k nearest neighbors'''
    distances = npl.norm(XTrain - Xvector, axis = 1) #create an array of dsitances between our vector and the training data
    neighbor_index = select_neighbor_index(distances, k) #find the index of the k enarest neighbors
    neighbors = YTrain[neighbor_index] #calculate the Y values of the k nearest neighbors
    if sum(neighbors) > k/2: #if the majority of k values are 1, predict 1 else predict 0
        return 1
    else:
        return 0

def select_neighbor_index(distances, k):
    '''Take in a vector of distances and calculate the k nearest neighbor indexes for them'''
    sorted_distances = np.sort(distances) #create a sorted array
    neighbor_distances = sorted_distances[:k] #take the first k values
    neighbor_index = []
    for i in neighbor_distances: #calculate the indexes for the first k values
        ind = np.where(distances == i)[0][0]
        neighbor_index.append(ind)
    return neighbor_index
    
def knn_predict(XTest, k = 1, XTrain = XTrain, YTrain = YTrain):
    '''Takes in an array of 13 values per row, and returns an array of predictions'''
    Ypred = []
    for i in XTest:
        Ypred.append(predict_one(i, k, XTrain, YTrain))
    Ypred = np.array(Ypred)
    return Ypred


# In[3]:


from sklearn.metrics import accuracy_score
# given classifier called knn , compute the accuracy on the testset
accTestTrain = accuracy_score(YTrain, knn_predict(XTrain))
accTestTest = accuracy_score(YTest, knn_predict(XTest))
print("Accuracy on Training Data: " + str(accTestTrain))
print("Accuracy on Testing Data: " + str(accTestTest))


# In[4]:


#Exercise 2
from sklearn.model_selection import KFold

def cross_validate(k, XTrain, YTrain):
    '''For a value k, perform 5 fold cross validation for k nearest neighbors'''
    loss_list = []
    # create indices for CV
    cv = KFold (n_splits = 5)
    # loop over CV folds
    for train, test in cv.split(XTrain):
        XTrainCV, XTestCV, YTrainCV, YTestCV = XTrain[train], XTrain[test], YTrain[train], YTrain[test]
        lossTest = loss(YTestCV, knn_predict(XTestCV, k, XTrainCV, YTrainCV))
        loss_list.append(lossTest)
    average_loss = np.mean(loss_list)
    return average_loss

def loss(true, predict):
    '''Takes in a true array and a predicted array and returns the loss array'''
    difference = (predict - true) ** 2 # square it to get positive values
    loss = sum(difference)/len(difference)
    return loss

def find_best_k(k_list, XTrain=XTrain, YTrain=YTrain):
    '''Given a list of ks, perform 5 fold cross validation for each k and return the best k'''
    k_loss = []
    for k in k_list:
        loss = cross_validate(k, XTrain, YTrain)
        k_loss.append(loss)
        print("Loss for "+ str(k) + " neighbors: " + str(loss))
    ind = k_loss.index(min(k_loss))
    best_k = k_list[ind]
    print("Best k: " + str(best_k))
    return best_k

k_list = [1,3,5,7,9,11]
find_best_k(k_list)


# In[5]:


#Exercise 3
best_acc_train = accuracy_score(YTrain, knn_predict(XTrain, k = 3))
best_acc_test = accuracy_score(YTest, knn_predict(XTest, k = 3))
print("Accuracy on Training Data: " + str(best_acc_train))
print("Accuracy on Testing Data: " + str(best_acc_test))


# In[6]:


#Exercise 4
def normalize_data(source, target):
    '''Takes in a source array and a target array. Normalizes the target array based on the source array'''
    mean = np.mean(source, axis = 0)
    sd = np.std(source, axis = 0)
    target = (target - mean)/sd
    return target

XTrain_norm = normalize_data(XTrain, XTrain)
XTest_norm = normalize_data(XTrain, XTest)

k_list = [1,3,5,7,9,11]
k_best = find_best_k(k_list, XTrain=XTrain_norm)
best_acc_train_norm = accuracy_score(YTrain, knn_predict(XTrain_norm, k = k_best, XTrain=XTrain_norm))
best_acc_test_norm = accuracy_score(YTest, knn_predict(XTest_norm, k = k_best, XTrain=XTrain_norm))
print("Accuracy on Training Data: " + str(best_acc_train_norm))
print("Accuracy on Testing Data: " + str(best_acc_test_norm))

