#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 11:47:31 2022

@author: gavinkoma
"""
import numpy as np
import matplotlib.pyplot as plt


#f(x) = 3x1^2 + 2x2^2 + 4x1x2 - 5x1 + 6
# derive analytically the values of x* that could satisfy the gradient
# okay so probably we want to take the partial derivation first
# maybe we start by defining our function

#f_x = 2*x_1**2 + 2*x_2**2 + 4*x_1*x_2 - 5*x_1 + 6
#f_xx = 10*x_1 + 8*x_2 - 5


#%% ## question 2
#You are given a function f(x) = sin(x) +0.3x
x_val = [*range(0,100)]
x = np.arange(-10,10,0.1)
y = np.sin(x)+0.3*x
plt.plot(x,y)
plt.show()

#derive a gradient descent iteration formula for finding a (local) minimum of f(x)
#we have to repeat our functions until we find convergence
y_dev = np.cos(x) + 0.3
x_dev = np.arange(-10,10,0.1)
plt.plot(x_dev,y_dev)
plt.show()

cur_x = [-0] #algorithm will start at 3
rate = 0.01 #learning rate
alpha = rate
max_iters = 1000 #maximum number of iterations

def gradient(val,alpha,max_iters):
    
    precision = 0.000001 #lets stop the algorithm here
    previous_step_size = 1
    
    iters = 0
    df = lambda x: np.cos(x)+0.3 #gradient
    
    while previous_step_size > precision and iters < max_iters:
        prev_x = val #store the current x value in prev_x
        val = val-rate*df(prev_x) #grad descent
        previous_step_size = abs(val - prev_x)
        iters = iters+1
        print("Iteration",iters,"\nX value is",val)
    
    return

for val in cur_x:
    gradient(val, alpha, max_iters)



