#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 18:00:06 2022

@author: gavinkoma
"""

import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras import datasets, layers, models
from keras.utils import np_utils
from keras import regularizers
from keras.layers import Dense, Dropout, BatchNormalization
import matplotlib.pyplot as plt
import numpy as np


(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Creating a list of all the class labels
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Visualizing some of the images from the training dataset
plt.figure(figsize=[10,10])
for i in range (25):    # for first 25 images
  plt.subplot(5, 5, i+1)
  plt.xticks([])
  plt.yticks([])
  plt.grid(False)
  plt.imshow(train_images[i], cmap=plt.cm.binary)
  plt.xlabel(class_names[train_labels[i][0]])

plt.show()

train_images=train_images.astype('float32')
test_images=test_images.astype('float32')

#standardize your images
train_images = train_images/255
test_images = test_images/255

#one hot encoding for our target class
num_classes = 10
train_labels = np_utils.to_categorical(train_labels,num_classes)


model = Sequential()

#im not sure how many layers of this we actually want 
#so we can try to make one and test out the accuracy and then just
#progress from there? 







