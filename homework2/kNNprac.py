# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets

#import the data first
iris = datasets.load_iris()


#pseudocode 
# predictor = create(algorithm_type, D)
# y_new = predictor(x_new)

#why dont we start by initially using the first two features
#and we can define our k value
k = 10
x = iris.data[:,[0,1]]
y = iris.target


print(iris.target)




