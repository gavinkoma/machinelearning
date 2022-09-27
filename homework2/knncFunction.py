#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 17:28:38 2022

@author: gavinkoma
"""
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from collections import Counter


#okay we need to have our function match the following form
#accuracy = knnC(X_train, X_test, y_train, y_test,k)
#recall that we are assuming euclidean distance again
iris = datasets.load_iris()

X,y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.33)

k_vals = [1,3,5,15,25,30]

k = 3



def euclidean(x1,x2):
    eu_distance = np.sqrt(np.sum((x1-x2)**2))
    return eu_distance


def knnC(X_train,X_test,y_train,y_test,k):
    
    
    fuck
    return

