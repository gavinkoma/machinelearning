#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 23:41:19 2022

@author: gavinkoma
"""

import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

#generate N=100 points from gaussian distribution with a mean
#of 2 and a standard deviation of 2
#plot a histogram of D

nums = []
mu = 2
sigma = 2

for i in range(100):
    temp = random.gauss(mu,sigma)
    nums.append(temp)
    
plt.hist(nums,bins=100)
plt.show()


#print the mean and the standard deviation of this sample
means = sum(nums)/len(nums)
print(means)
std = np.std(nums)
print(std)

#calculate the log-likelihood of varying sigma/mu values





