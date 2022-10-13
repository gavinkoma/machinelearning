#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 11:47:31 2022

@author: gavinkoma
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#f(x) = 3x1^2 + 2x2^2 + 4x1x2 - 5x1 + 6
# derive analytically the values of x* that could satisfy the gradient
# okay so probably we want to take the partial derivation first
# maybe we start by defining our function

#f_x = 2*x_1**2 + 2*x_2**2 + 4*x_1*x_2 - 5*x_1 + 6
#f_xx = 10*x_1 + 8*x_2 - 5


## question 2
#You are given a function f(x) = sin(x) +0.3x
x_val = [*range(0,100)]
x = np.arange(-10,10,0.1)
y = np.sin(x)+0.3*x
plt.plot(x,y)
plt.show()

#derive a gradient descent iteration formula for finding a (local) minimum of f(x)
#we have to repeat our functions until we find convergence
x_dev = np.cos(x) + 0.3
y_dev = np.arange(-10,10,0.1)
plt.plot(x_dev,y_dev)
plt.show()

def mean_squared_error


