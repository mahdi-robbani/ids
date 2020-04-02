#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import everything
import numpy as np
import linreg as lin
from sklearn.ensemble import RandomForestClassifier
import gradientdescent as gd
import logreg as log
import cluster as cl
import pca
import matplotlib.pyplot as plt
from plot import *


# In[2]:


# Exercise 2
#load data
wine_train = np.loadtxt("data/redwine_training.txt")
names = ["Intercept", "Fixed acidity", "Volatile acidity","Citric acid","Residual sugar","Chlorides","Free sulfur dioxide",
         "Total sulfur dioxide", "Density","pH","Sulfates","Alcohol"]
#separate data
wine_x_train = wine_train[:, :-1]
wine_x_train = lin.normalize_data(source=wine_train[:, :-1], target=wine_x_train)
wine_x_train_col1 = wine_x_train[:, :1]
wine_y_train = wine_train[:, -1:]

# Exercise 2b
weights_one = lin.multivarlinreg(wine_x_train_col1, wine_y_train)
print("Weights for first feature:")
lin.print_names(weights_one, names)

#Exercise 2c
weights_full = lin.multivarlinreg(wine_x_train, wine_y_train)
print("Weights for all features:")
lin.print_names(weights_full, names)


# In[3]:


# Exercise 3
#load data
wine_test = np.loadtxt("data/redwine_testing.txt")
#separate data
wine_x_test = wine_test[:, :-1]
wine_x_test = lin.normalize_data(source=wine_train[:, :-1], target=wine_x_test)
wine_x_test_col1 = wine_x_test[:, :1]
wine_y_test = wine_test[:, -1:]

# Exercise 3b
rmse_one = lin.test(weights_one, wine_x_test_col1, wine_y_test)
print(f"RMSE for first feature: {rmse_one}")

# Exercise 3c
rmse_full = lin.test(weights_full, wine_x_test, wine_y_test)
print(f"RMSE for all features: {rmse_full}")


# In[4]:


# Exercise 5
# read in the data
weed_train = np.loadtxt('data/IDSWeedCropTrain.csv', delimiter = ',')
weed_test = np.loadtxt('data/IDSWeedCropTest.csv', delimiter = ',')

# split input variables and labels
weed_x_train = weed_train[:,:-1]
weed_y_train = weed_train[:,-1]
weed_x_test = weed_test[:,:-1]
weed_y_test = weed_test[:,-1]

#create forest
RF = RandomForestClassifier(n_estimators=50)
RF = RF.fit(weed_x_train, weed_y_train)
RF_score = RF.score(weed_x_test, weed_y_test)
print(f" Accuracy of Random Forest: {RF_score}")


# In[5]:


# Exercise 6
x = np.linspace(-1.5, 1.5, 100)
rates = [0.1, 0.01, 0.001, 0.0001]
iterations = []
function_values = []

for i in rates:
    i, y = gd.gradient_descent(x, i)
    iterations.append(i)
    function_values.append(y)
    
for i in range(len(rates)):
    print(f"Leaning Rate: {rates[i]}, Iteration: {iterations[i]}, Value: {function_values[i]}")


# In[6]:


# Exercise 7
Iris2D1_train = np.loadtxt("data/Iris2D1_train.txt")
Iris2D1_test = np.loadtxt("data/Iris2D1_test.txt")
Iris2D2_train = np.loadtxt("data/Iris2D2_train.txt")
Iris2D2_test = np.loadtxt("data/Iris2D2_test.txt")

# Exercise 7.1
log.plot_iris(Iris2D1_train, "Iris2D1 train")
log.plot_iris(Iris2D1_test, "Iris2D1 test")
log.plot_iris(Iris2D2_train, "Iris2D2 train")
log.plot_iris(Iris2D2_test, "Iris2D2 test")


# In[7]:


# Exercise 7.3
e1, w1, p1 = log.log_regression(Iris2D1_train, Iris2D1_test)
e2, w2, p2 = log.log_regression(Iris2D2_train, Iris2D2_test)
# Exercise 7.4
print(f"Iris2D1 Error: {e1}\nIris2D1 Weights:\n{w1}")
print("\n")
print(f"Iris2D2 Error: {e2}\nIris2D2 Weights:\n{w2}")


# In[8]:


# Exercise 9
# load data
digits = np.loadtxt("data/MNIST_179_digits.txt")
labels = np.loadtxt("data/MNIST_179_labels.txt")

# Exercise 9a
# get proportions and pictures
numbers = [1,7,9]
cl.get_prop_plot(3, numbers, digits, labels, "Raw data Cluster ")


# In[9]:


# Exercise 9b
k_list = [1, 3, 5, 7, 9, 11]
cl.find_best_k(k_list, digits, labels)


# In[10]:


# Exercise 10
# Exercise 10a
e_vec, e_val = pca.pca(digits)
cum_variance = np.cumsum(e_val/sum(e_val)) * 100
count = list(range(1, len(e_val)+1))
plt.plot(count, cum_variance)
plot_template(title= 'Cumulative Variance versus Principal Components of Digits',
             xlabel='Number of Principcal Components',
             ylabel='Percentage of Variance Captured',
             equal_axis=False, save = False)


# In[11]:


# Exercise 10b
dimensions = [1, 20, 200]

for i in dimensions:
    cl.get_prop_plot_mds(3, numbers, digits, labels, i, " dimensions Cluster ")


# In[12]:


# Exercise 10c
for i in dimensions[1:]:
    cl.mds_acc(i, k_list, digits, labels)

