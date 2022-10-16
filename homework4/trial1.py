#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 11:47:31 2022

@author: gavinkoma
"""
import numpy as np
import matplotlib.pyplot as plt

#%%
#question 1
#begin by defining our two partials
def derivativex1(x1,x2):
    return 6*x1+4*x2-5

def derivativex2(x1,x2):
    return 4*x2+4*x1

alpha = [0.00001,0.0001,0.001,0.01,0.1] #define our values for alpha

def gradient(x0,alpha,iter):
    evolutionx = [] #list to append our evolution of x value
    lossval = [] #list to append our loss value
    
    for zeus in range(iter):
        evolutionx.append([x0[0],x0[1]])
        prev_x0_1 = x0[0] #define the previous variables
        prev_x0_2 = x0[1] #define the previous variable again (x_2)
        x0[0]= x0[0] - alpha * derivativex1(prev_x0_1,prev_x0_2) #our new value is simga - alpha*previous rate in equation 1
        x0[1]= x0[1] - alpha * derivativex2(prev_x0_1,prev_x0_2) #the next value is defined by the same equation but for x[1] not x[0]
        #we have to make two separate equations because if we dont then we cant calculate the partials, we would be calculating the gradiend
        
        #prev is the original untouched equation athat we want to pass our x1, x2 values to
        prev = 3*prev_x0_1**2 + 2*prev_x0_2**2 + 4*prev_x0_1*prev_x0_2 - 5*prev_x0_1 + 6
        diff = abs(prev - (3*x0[0]**2 + 2*x0[1]**2 + 4*x0[0]*x0[1] - 5*x0[0] + 6)) #difference calculates the differences 
        #between our x_0 & x_1 as they are updated through our functions
        #the difference between these values defines our loss
        lossval.append(diff)
        
           
        #the degree to which we should calculate loss (4 decimals), could be smaller 
        if diff < 0.0001:
            return evolutionx, lossval           
    #return values
    return evolutionx, lossval     

#set up a list for our iteration values 
#we cannot run larger than 0.1 because if results in matrices that are too large 
#to process, spoke with Dr. Vucetic about it, he is okay with stopping at 0.1
alpha = [0.00001, 0.0001, 0.001, 0.01, 0.1]

#define a place to store results and loss
res = {}
loss = {}

#loop through our functions start at index 0, first alpha iteration and 1000 iters
for a in alpha:
    xfinal, loss_final = gradient([0,0],a,1000)
    #res[alpha iteration] = final
    res[a] = xfinal
    #loss[alpha iteration]=loss 
    loss[a] = loss_final

#plot our graphs to get a view of how our gradient descent looks:
for i in res:
    xs = [x[0] for x in res[i]]
    ys = [x[1] for x in res[i]]
    plt.plot(xs,ys)


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

cur_x = [10] #algorithm will start at 3
alpha = [0.01] #learning rate
max_iters = 2000 #maximum number of iterations

def gradient(val,alpha,max_iters):
    
    precision = 0.000001 #lets stop the algorithm here
    previous_step_size = 1
    
    iters = 0
    df = lambda x: np.cos(x)+0.3 #gradient
    
    iter_val = []
    
    while previous_step_size > precision and iters < max_iters:
        prev_x = val #store the current x value in prev_x
        val = val-alpha*df(prev_x) #grad descent
        previous_step_size = abs(val - prev_x)
        iters = iters+1
        print("Iteration",iters,"\nX value is",val)
        
        iter_val.append(val)
    
    plt.figure()
    plt.scatter(range(len(iter_val)), iter_val)
    plt.ylim(max(iter_val), min(iter_val))
    plt.title("Iteration Progression")
    plt.xlabel("Iteration Progression")
    plt.ylabel("Local Minima Value Progression")
    
    return 

for val in cur_x:
    for a_val in alpha:
       gradient(val, a_val, max_iters)
    
    
#%% question 3

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification, load_iris
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis



iris = load_iris()

h = 0.02 #keep the same step size in our mesh

names = ["Nearest Neighbors", "Linear SVM","RBF SVM",
         "Decision Tree","Random Forest","Neural Net","AdaBoost"]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel = "linear",C=0.025),
    SVC(gamma=2,C=1),
    #GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5,n_estimators=10,max_features=1),
    MLPClassifier(alpha=1),
    AdaBoostClassifier()
    ]

scores = []


xtrain,xtest,ytrain,ytest = train_test_split(iris.data,iris.target,train_size=.75)


# iterate over classifiers

for name, clf in zip(names, classifiers):
    clf.fit(xtrain, ytrain)
    score = clf.score(xtest, ytest)
    scores.append(score)
    
print(scores)
    












































