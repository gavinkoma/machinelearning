#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 11:47:31 2022

@author: gavinkoma
"""
import numpy as np
import matplotlib.pyplot as plt


#%% question 1
#f(x) = 3x1^2 + 2x2^2 + 4x1x2 - 5x1 + 6
# derive analytically the values of x* that could satisfy the gradient
# okay so probably we want to take the partial derivation first
# maybe we start by defining our function

#f_x = 2*x_1**2 + 2*x_2**2 + 4*x_1*x_2 - 5*x_1 + 6
#f_xx = 10*x_1 + 8*x_2 - 5

def derivative_x1(x1,x2):
    return 6*x1+4*x2-5

def derivative_x2(x1,x2):
    return 4*x2+4*x1

def gd_hw3_1(x0,alpha,iter):
    evolution = list()
    loss = list()
    for i in range(iter):
        evolution.append([x0[0],x0[1]])
        prev_x0_1 = x0[0]
        prev_x0_2 = x0[1]
        x0[0]= x0[0] - alpha * derivative_x1(prev_x0_1,prev_x0_2)
        x0[1]= x0[1] - alpha * derivative_x2(prev_x0_1,prev_x0_2)
        prev = 3*prev_x0_1**2 + 2*prev_x0_2**2 + 4*prev_x0_1*prev_x0_2
        diff = abs(prev - (3*x0[0]**2 + 2*x0[1]**2 + 4*x0[0]*x0[1] - 5*x0[0] + 6))
        loss.append(diff)
        
        if diff < 0.0001:
            return evolution, loss
        
    return evolution, loss


alpha = [0.00001, 0.0001, 0.001, 0.01, 0.1]
res = dict()
loss = dict()

for a in alpha:
    xfinal, loss_final = gd_hw3_1([0,0],a,1000)
    res[a] = xfinal
    loss[a] = loss_final


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
    












































