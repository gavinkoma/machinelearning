#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 15:59:35 2022

@author: gavinkoma
"""
#%% start by importing the proper libraries that youll be needing
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from collections import Counter

#repeat question 9
#train predictor using differenct choices of k. try k = 1,3,5,15,25,30
#Report accuracies on the test data (you can use the score method).
#Which choice of k resulted in the highest accuracy? Comment briefly if the
#results make sense to you. 

def euclideandistance(x1,x2):
    return np.sqrt(np.sum((x1-x2)**2))

def knnC(X_train, X_test, y_train, y_test, k):
    #lets first define our values
    op_labels = []
    
    #start with looping through data points
    for item in X_test:
        
        #dont forget that you need to store distances somewhere
        point_dist = []
        
        #loop through the training data
        for val in range(len(X_train)):
            distances = euclideandistance(np.array(X_train[val,:]),item)
            point_dist.append(distances)
        point_dist = np.array(point_dist)
        
        
        #sort the array
        dist = np.argsort(point_dist)[:k]
        
        #find the labels
        labels = y[dist]
        
        #find the majority of the labels
        lab = Counter(labels).most_common(1)
        op_labels.append(lab)
    return op_labels

# for k in k_vals:
#     knnC(X_train,X_test,y_train,y_test,k)


#and we also need to test our data on some stuff
#which we will use our X_test y_test stuff for

iris = datasets.load_iris()
X,y = iris.data, iris.target

k_vals = [1,3,5,15,25,30]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33)

#accuracy = knnC(X_test,y_test,X_train,y_train,k) where k is the nearest neighbors


k = 3

op_labels = knnC(X_train,X_test,y_train,y_test,k)














