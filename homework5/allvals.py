#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 15:09:40 2022

@author: gavinkoma
"""
#start with the proper libraries i guess
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.utils import np_utils
from tensorflow.keras.optimizers import Adam
from keras.datasets import mnist
from sklearn.model_selection import KFold
from matplotlib import pyplot as plt



(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_test_orig = X_test

#recall that we want all 10 digits
#lets convert y_train and y_test to categorical 

X_train = X_train.reshape((X_train.shape[0],28,28,1))
X_test = X_test.reshape((X_test.shape[0],28,28,1))

#convert to categorical
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

#now we need to normalize the pixel values
train_norm = X_train.astype('float32')
test_norm = X_test.astype('float32')

train_norm = train_norm/255.0
test_norm = test_norm/255.0


nb_epoch = 5  # kept very low! Please increase if you have GPU

batch_size = 128
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 3

#we previously designated 7 as 1 and everything else as 0
#via binary classification, but we dont need to do this
#because were using all the numbers, so Im just going to skip it

model = Sequential()
model.add(Conv2D(32, (3,3), activation = 'relu',kernel_initializer='he_uniform',input_shape=(28,28,1)))
model.add(MaxPooling2D((2,2)))
model.add(Flatten())
model.add(Dense(100,activation='relu',kernel_initializer='he_uniform'))
model.add(Dense(10,activation='softmax'))

#compile it
opt = Adam(learning_rate=0.0002, amsgrad=True)
model.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['accuracy'])

#i guess we can use cross validation but?
#im not entirely sure? it would work to keep the eval times low
#since our data sets wont be quite small 

scores, histories = [],[] 


kfold = KFold(5,shuffle=True,random_state=1)
#enumerate baybeeeee

model.summary()

# Evaluating the model on the test data    
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print('Test Loss:', loss)
print('Test Accuracy:', accuracy)

def show_results(model):
    rows = 4
    columns = 15
    sliced = rows*columns
    predicted = model.predict(X_test[:sliced]).argmax(-1)

    plt.figure(figsize=(16,8))
    for i in range(sliced):
        plt.subplot(rows, columns, i+1)
        plt.imshow(X_test_orig[i], cmap='gray', vmin=0, vmax=255)
        color = 'black' if y_test[i, 1] == predicted[i] else 'red'
        plt.text(0, 0, predicted[i], color=color, 
                 bbox=dict(facecolor='white', alpha=1))
        plt.axis('off')

show_results(model)

