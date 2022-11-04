#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 15:09:40 2022

@author: gavinkoma
"""
#start with the proper libraries i guess
from tensorflow.keras.utils import to_categorical
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
import numpy as np
import matplotlib.pyplot as plt

#from step 0
nb_epoch = 5
batch_size = 64
nb_filters = 32
nb_pool = 2
nb_conv = 3


(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_test_orig = X_test
img_rows, img_cols = 28, 28
shape_ord = (img_rows,img_cols,1)

#normalization
X_train = X_train[:20000]
Y_train = y_train[:20000]

idx = np.random.permutation(len(X_train))
X_train,Y_train = X_train[idx],Y_train[idx]

X_train = X_train.reshape((X_train.shape[0],) + shape_ord)
X_test = X_test.reshape((X_test.shape[0],) + shape_ord)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train/255
X_test = X_test/255
Y_train = to_categorical(Y_train)
Y_test = to_categorical(y_test)

nb_classes = 10


# Function for constructing the convolution neural network
# Feel free to add parameters, if you want

def build_model(num_conv = 1, conv_activation = "relu", num_dense = 1, dense_activation  = "relu", 
               dropout = True, max_pooling = True):
    """"""
    model = Sequential()
    model.add(Conv2D(nb_filters, (nb_conv, nb_conv), padding='valid', input_shape=shape_ord))
    model.add(Activation(conv_activation))
    model.add(Conv2D(nb_filters, (nb_conv, nb_conv)))
    model.add(Activation(conv_activation)) 
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dropout(0.50))
    model.add(Dense(128))
    model.add(Activation(dense_activation))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
    model.fit(X_train, Y_train, batch_size=batch_size, 
              epochs=nb_epoch,verbose=1,
              validation_data=(X_test, Y_test))
          

    #Evaluating the model on the test data    
    score, accuracy = model.evaluate(X_test, Y_test, verbose=0)
    print(num_conv , 'convolutional layers,', num_dense, "dense layers")
    if max_pooling: print("With max pooling")
    if dropout: print("With dropout")
    print('Test score:', score)
    print('Test accuracy:', accuracy)
    print(model.summary())
    show_results(model)
    #return accaccuracy

def show_results(model):
    rows = 4
    columns = 15
    sliced = rows*columns
    predicted = model.predict(X_test[:sliced]).argmax(-1)
    plt.figure(figsize=(16,8))
    for i in range(sliced):
        plt.subplot(rows, columns, i+1)
        plt.imshow(X_test_orig[i], cmap='gray', vmin=0, vmax=255)
        color = 'black' if np.where(Y_test[i] == 1)[0][0] == predicted[i] else 'red'
        plt.text(0, 0, predicted[i], color=color, 
                 bbox=dict(facecolor='white', alpha=1))
        plt.axis('off')


build_model(num_conv = 3, num_dense = 2)


