#!/usr/bin/env python
# coding: utf-8

# In[194]:


# Exercise 6
import numpy as np
import matplotlib.pyplot as plt
from plot import *

def f(x):
    f = np.exp(-0.5*x) + 10*x**2
    return f

def df(x):
    d = -0.5*np.exp(-0.5*x) + 20*x
    return -d

def tangent(x, p):
    m = df(p)
    c = f(p)
    return -m*(x-p) + c

def gradient_descent(x, learning_rate):   
    #settings
    old_x = 1 # start
    iteration = 1
    tolerance = 10**-10
    
    #plot
    y = f(x)
    fig, (ax1, ax2) = plt.subplots(2)
    fig.suptitle("Learning rate = " + str(learning_rate))
    ax1.plot(x, y, label = "f(x)")
    ax2.plot(x, y, label = "f(x)")
    
    while True:
        m = df(old_x)
        new_x = old_x + m*learning_rate
        difference = abs(new_x - old_x)
        
        if iteration < 4:
            line = tangent(x, old_x)
            ax1.plot(x, line, color = 'C'+str(iteration))
            ax1.scatter(old_x, f(old_x), label = "Iter = " + str(iteration), color = 'C'+str(iteration), zorder = 3, ec ="black")
            ax2.scatter(old_x, f(old_x), label = "Iter = " + str(iteration), color = 'C'+str(iteration), zorder = 3, ec ="black")
        elif iteration < 11:
            ax2.scatter(old_x, f(old_x), label = "Iter = " + str(iteration), color = 'C'+str(iteration), zorder = 3, ec ="black")
        elif iteration == 10000:
            print("Max iterations reached")
            break
        elif difference < tolerance:
            print("Tolerance reached")
            break
        elif np.isinf(m):
            print("Infinite Gradient")
            break
        
        old_x = new_x
        iteration += 1
    
    #plot details
    ax1.grid()
    ax2.grid()
    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    box = ax2.get_position()
    ax2.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax2.legend(loc='lower left', bbox_to_anchor=(1, 0.5), shadow = True)
#     fig.savefig("Learning rate = " + str(learning_rate) + ".png", dpi = 200)
#     fig.clf()
    plt.show()
    
    old_y = f(old_x)
    print(f"X value = {old_x}")
    print(f"Y value = {old_y}")
    print(f"Gradient = {m}")
    print(f"Iteration = {iteration}")
    
    return iteration, old_y


# In[195]:


x = np.linspace(-1.5, 1.5, 100)
rates = [0.1, 0.01, 0.001, 0.0001]

for i in rates:
    gradient_descent(x, i)
