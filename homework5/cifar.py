#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 18:00:06 2022

@author: gavinkoma
"""
#start with the proper libraries i guess
from tensorflow.keras.utils import to_categorical
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
import numpy as np
import matplotlib.pyplot as plt

#load your data
(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
X_test_orig = X_test
img_rows = 32
img_cols = 32
shape_ord = (img_rows,img_cols,3)


#from step 0, just going to keep using these for now
nb_epoch = 5
batch_size = 64
nb_filters = 32
nb_pool = 2
nb_conv = 3
nb_classes = 10

#normalization brother
X_train = X_train[:20000]
Y_train = Y_train[:20000]
X_train = X_train.reshape((X_train.shape[0],) + shape_ord)
X_test = X_test.reshape((X_test.shape[0],) + shape_ord)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train = X_train/255
X_test = X_test/255

Y_train = to_categorical(Y_train)
Y_test = to_categorical(Y_test)


def show_results(model):
    rows = 4
    columns = 15
    sliced = rows*columns
    predicted = model.predict(X_test[:sliced]).argmax(-1)
    plt.figure(figsize=(16,8))
    
    for i in range(sliced):
        plt.subplot(rows,columns,i+1)
        plt.imshow(X_test_orig[i],cmap='gray',vmin=0,vmax=255)
        color = 'black' if np.where(Y_test[i] ==1)[0][0] == predicted[i] else 'red'
        plt.text(0,0,predicted[i],color = color,
                 bbox=dict(facecolor='white',alpha = 1))
        plt.axis('off')


#lets build the model
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
    model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])
    model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch,verbose=1,validation_data=(X_test, Y_test))
    
    score, accuracy = model.evaluate(X_test, Y_test, verbose = 0)
    print(num_conv, 'convolutional layers,', num_dense, "dense layers")
    if max_pooling: print("with max pooling")
    if dropout: print("with dropout")
    
    print('Test score: ' + str(score))
    print('Test accuracy: ' +str(accuracy))
    print(model.summary())
    print(model.summary)
    show_results(model)
    return

        
build_model(num_conv=3, num_dense=2)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
              
              
              
              
              
              
              
              