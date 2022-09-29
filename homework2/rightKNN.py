#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 21:33:01 2022

@author: gavinkoma
"""

#okay start by importing the right libraries 
import numpy as np
from scipy.stats import mode
from sklearn import datasets
from sklearn.model_selection import train_test_split
from collections import Counter


#okay so we cant use the KNN actual functions but we can use it to split our data
#because we do still need our training and our test sets
#lets define our data

iris = datasets.load_iris()
X,y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.33)
k_vals = [1,3,5,15,25,30]

#okay so we have to define a distance function that will calculate for us
#remember that we are using euclidean distance here and that
#we will need to pass two values to it for our calculations
def euclideandistance(x1,x2):
    #square root of sum of difference squared
    return np.sqrt(np.sum((x1-x2)**2)) #condense instead of making another variable


#we will also need a function to define the accuracy of our model
def accuracy(y_test,y):
    return (np.sum(y_test==y)/len(y))
    

def knnC(X_train,X_test,y_train,y_test,k):
    #create a list for our predicted labels this will hold the vals later
    predicted_labels = []
    #iterate through X_test each value (we need to calculate the distance)
    for xtest in X_test:
        #we also need a distance list to append to
        pdis = []
        #for every xtest value in the total length/range of x_train we are going
        #to calculate our distance and then append it to our distance list
        for val in range(len(X_train)):
            pdis.append(euclideandistance(xtest,np.array(X_train[val,:])))

        #sort through the dist values and slice them according to our 
        #variable k number
        dist = np.argsort(np.array(pdis))[:k]
        
        labels = y_train[dist]
        
        #take the majority vote
        lab = mode(labels)
        lab = lab.mode[0]
        predicted_labels.append(lab)
   
    return predicted_labels

#run the function like question 9 for varying k numbers
for val in k_vals:
    y_pred = knnC(X_train,X_test,y_train,y_test,val)
    acc = accuracy(y_test,y_pred)
    print("The accuracy for a k-value of " + str(val) + " is: " + str(acc))



