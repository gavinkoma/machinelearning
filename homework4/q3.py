#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 15 12:09:02 2022

@author: gavinkoma
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification, load_iris
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

data1 = pd.read_csv("adult.data",sep=',',header = None)
data2 = pd.read_csv("adult.names",sep=';',header = None)
data3 = pd.read_csv("adult.test",sep=',')

combined = [data1,data2,data3]
combined = pd.concat(combined)

header = ['age','workclass','fnlwgt','education','education-num','marital-status',
          'occupation','relationship','race','sex','capital-gain','capital-loss',
          'hours-per-week','native-country','income','cross/val']

combined.columns=header

# income_types = ('<=50K','>50K')
# income_df = pd.DataFrame(income_types,columns = ['income'])

# income_df['income'] = income_df['income'].astype('category')

# income_df['Income_Types_Cat'] = income_df['income'].cat.codes
# income_df
le = LabelEncoder()
combined.income = le.fit_transform(combined.income)

#data is entered and combined so life is good rn 
h = 0.02

names = ["Nearest Neighbors", "Linear SVM","RBF SVM",
         "Decision Tree","Random Forest","Neural Net","AdaBoost"]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
#    GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1),
    AdaBoostClassifier()]
#    GaussianNB(),
#    QuadraticDiscriminantAnalysis()]

#what do we want to look at? 
#we have so many values here and I dont think the goal is to look at them all 
#so maybe we should look at income??? 
#correlation plot?

correlation = combined.corr()
sns.heatmap(correlation,annot=True)



#our correlation graphs shows values for education num
#capital gain, capital loss, and hours per week
#so lets make other graphs that compare these with income? 

X,y = make_classification(n_features=2, n_redundant=0,n_informative=2,
                          random_state=1,n_clusters_per_class=1)

rng = np.random.RandomState(2)
X += 2 * rng.uniform(size=X.shape)
linearly_separable = (X,y)

#we should set our datasets now so we can use them for iterations later
educa_income = pd.DataFrame().assign(education=combined['education-num'], income=combined['income'])
gain_income = pd.DataFrame().assign(gain=combined['capital-gain'], income=combined['income'])
loss_income = pd.DataFrame().assign(loss=combined['capital-loss'], income=combined['income'])
hour_income = pd.DataFrame().assign(hour=combined['hours-per-week'], income=combined['income'])


datasets = [educa_income,gain_income,loss_income,hour_income]


figure = plt.figure(figsize = (30,10))

i = 1

for ds_cnt, ds in enumerate(datasets):
    # preprocess dataset, split into training and test part
    X, y = ds
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=.4, random_state=42)

    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # just plot the dataset first
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
    if ds_cnt == 0:
        ax.set_title("Input data")
    # Plot the training points
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)
    # and testing points
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6)
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    i += 1














