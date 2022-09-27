# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
cmap = ListedColormap(['#FF0000','#00FF00','#0000FF'])


#our sample is classified by the weight of the nearest neighbors
#lets do two classes to start and a feature vector
#how any nearest neighbors do we want to look at? 3? 4?
#remember that were using euclidean distance




class kNN:
    
    def __init__(self, k=3):
        self.k=k
        
    #so we now have our class
    
    def fit(self,X,y): #will probably be our training step
        pass
    
    def predict(self,X):
        pass
    




