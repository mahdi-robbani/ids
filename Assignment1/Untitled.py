#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Exercise 1
import numpy as np
data = np.loadtxt('smoking.txt')
nonsmokers = data[np.where(data[:,4] == 0)] #select all rows for nonsmokers
smokers = data[np.where(data[:,4] == 1)] #select all rows for smokers

def avg_FEV1(data):
    '''Calculates average FEV1 value for an array'''
    fev1 = data[:,1] #Select the column for fev1 values
    avg = np.mean(fev1)
    return avg

print("Average FEV1 for nonsmokers: " + str(avg_FEV1(nonsmokers)))
print("Average FEV1 for smokers: " + str(avg_FEV1(smokers)))


# In[ ]:


#Exercise 2
import matplotlib.pyplot as plt
nonsmokers_fev1 = nonsmokers[:,1] #select fev1 values from the nonsmoker array
smokers_fev1 = smokers[:,1] #select fev1 values from the smoker array
labels = ["Non-smokers", "Smokers"]
plt.boxplot([nonsmokers_fev1, smokers_fev1], labels = labels)
plt.title('Boxplot for FEV1 levels in nonsmokers and smokers')
plt.ylabel('FEV1 levels')
plt.show()

