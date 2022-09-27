# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
from collections import Counter
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
cmap = ListedColormap(['#FF0000','#00FF00','#0000FF'])


#our sample is classified by the weight of the nearest neighbors
#lets do two classes to start and a feature vector
#how any nearest neighbors do we want to look at? 3? 4?
#remember that were using euclidean distance

iris = datasets.load_iris()
X,y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.33, 
                                                    random_state=1234)

# plt.figure()
# #ok this plots the first two features
# plt.scatter(X[:,0],X[:,1],c=y,cmap=cmap,edgecolor='k',s=20)
# plt.show()

def euclidean_distance(x1,x2):
    return np.sqrt(np.sum((x1-x2)**2))


class kNN:
    
    def __init__(self, k=3):
        self.k=k
        
    #so we now have our class
    
    def fit(self,X,y): #will probably be our training step
        #we want to store our training samples here and use them later
        self.X_train = X
        self.y_train = y
        
    
    def predict(self,X):
        #this can receive multiple samples
        #so we can write a mini helper method
        predicted_labels = [self.predict(x) for x in X]
        return(np.array(predicted_labels))
    
    def _predict(self,x):
        #we want to compute euclidean distance
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        
        #get the k nearest neighbors samples, also get labels
        #sort the distances and return the indices of how this is sorted
        k_indices = np.argsort(distances)[:self.k] #we only want k closest #indices of the k nearest samples
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        
        #majority vote, most common class label
        #this will print a tuple with what is the most common item 
        #and the second item is the number of times that it is in our list
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

    

from kNNprac import kNN
clf = kNN(k=3)
clf.fit(X_train,y_train)
predictions = clf.predict(X_test)


acc = np.sum(predictions == y_test) / len(y_test)
print(acc)








