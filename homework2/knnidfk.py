#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 21:01:13 2022

@author: gavinkoma
"""

import numpy as np
from scipy.stats import mode
from sklearn import datasets
from sklearn.model_selection import train_test_split

def distance(test,train):
    return np.sqrt(np.sum((test-train)**2))

def accuracy(y_hat,y):
    return (np.sum(y_hat==y) / len(y))

def KnnC(X_test,X_train,y_train,y_test,k):
    #neighbors = list()
    pred_labels = list()
    for test in X_test:
        point_dist = list()
        for j in range(len(X_train)):
            point_dist.append(distance(test,np.array(X_train[j,:])))
            
        point_dist = np.array(point_dist)
        
        dist = np.argsort(point_dist)[:k]
        
        labels = y_train[dist]

        lab = mode(labels)
        lab = lab.mode[0]
        pred_labels.append(lab)
    return pred_labels        


iris = datasets.load_iris()

X,y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.33)

y_pred = KnnC(X_test,X_train,y_train,y_test,1)
acc = accuracy(y_test, y_pred)
print(acc)




