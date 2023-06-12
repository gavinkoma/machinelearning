#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 23:41:19 2022

@author: gavinkoma
"""

import random
import numpy as np
import matplotlib.pyplot as plt
import math

#%% part a
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


#%%part b
#calculate the log-likelihood of varying sigma/mu values

ground_truth = 100

def log_likelihood(mu,sigma,nums):
    log_like = np.sum(math.log(2*math.pi*(sigma**2))/2+((nums-mu)**2)/(2*(sigma**2))) 
    return log_like
    
mus = [1,1.5,2]
sigs = [2,2,2]
max_log_like = 0
mu = ""
sigma = ""

for i,j in zip(mus,sigs):
    log_like = log_likelihood(i, j, ground_truth)
    if log_like > max_log_like:
        max_log_like = log_like
        mu = i
        sigma = j
        
print("the highest log_likelihood is: ",max_log_like, "with mu {} and sigma {}".format(mu,sigma))


#%%part c
#find the maximum likelihood estimation of u,sigma; how does this compare to the true parameters

from scipy import stats
from scipy.optimize import minimize

np.random.seed(1)
n=100

sample_data = np.random.normal(loc=2,scale=2,size=n)

def gaussian(params):
    mean=params[0]
    sd = params[1]
    
    #negative log likelihood?
    nll=-np.sum(stats.norm.logpdf(sample_data,loc=mean,scale=sd))
    return nll

initParams=[1,1]
results = minimize(gaussian,initParams,method='Nelder-Mead')
print(results.x)


#%%part d 1

lst=list(range(1,201))
count = 0

arr1 = list()


for val in lst:
    if count<=50:
        n=100
        
        sample_data = np.random.normal(loc=2,scale=2,size=n)
        
        def gaussian(params):
            mean=params[0]
            sd=params[1]
            
            #calculate nll
            nll=-np.sum(stats.norm.logpdf(sample_data,loc=mean,scale=sd))
            return nll
        
        initParams=[1,1]
        results=minimize(gaussian,initParams,method='Nelder-Mead')

        count+=1 
        #arr = np.ndarray(results.x)
        
        arr1.append((results.x[0], results.x[1]))
        print(results.x)
        
import pandas as pd
#this is our values for our optimization 
arr = pd.DataFrame(arr1, columns=['mean', 'stdev'])
print(arr)
  
plt.hist(arr['mean'],bins=100)
plt.hist(arr['stdev'],bins=100)
plt.legend(['Mean','Standard Deviation'])
plt.show()


#%%part d2 all 200 values

lst=list(range(1,201))
count = 0

arr1 = list()


for val in lst:
    if count<=201:
        n=100
        
        sample_data = np.random.normal(loc=2,scale=2,size=n)
        
        def gaussian(params):
            mean=params[0]
            sd=params[1]
            
            #calculate nll
            nll=-np.sum(stats.norm.logpdf(sample_data,loc=mean,scale=sd))
            return nll
        
        initParams=[1,1]
        results=minimize(gaussian,initParams,method='Nelder-Mead')

        count+=1 
        #arr = np.ndarray(results.x)
        
        arr1.append((results.x[0], results.x[1]))
        print(results.x)
        
import pandas as pd
#this is our values for our optimization 
arr = pd.DataFrame(arr1, columns=['mean', 'stdev'])
print(arr)
  
plt.hist(arr['mean'],bins=100)
plt.hist(arr['stdev'],bins=100)
plt.legend(['Mean','Standard Deviation'])
plt.show()

#the results begin to look more similar to that of a gaussian distribution
#when we increase the count number to 200 instead of 50. When it is left
#at 50, we see something more similar to that of a uniform distribution

#The Maximum Likelihood Estimation is an optimization algorithm that searches
#for the most suitable parameters. Since we know the data distribution is supposed
#to resemble that of a gaussian distribution, it makes sense that with more 
#values, we see something that closer resembles a gaussian distribution
#instead of that of a uniform distribution.







