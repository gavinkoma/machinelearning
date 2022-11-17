#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 10:45:02 2022

@author: gavinkoma
"""
#import libraries
from __future__ import print_function
from keras.callbacks import LambdaCallback
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from tensorflow.keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys
import io
import pandas as pd


#only need to run once, dont need to perform this again.
def datainit():
    #okay start by downloading data and reading through it
    start = pd.read_csv('TweetsElonMusk2010_2021.csv')
    new = pd.read_csv('rawdata2022.csv')
    
    tweets2010_2021 = start['tweet']
    tweets2022 = new['Tweets']
    
    #append the data to one list
    alltweets = pd.DataFrame(tweets2010_2021.append(tweets2022, 
                                                    ignore_index = True))
    
    #lets make it a text file? 
    with open('tweet.txt','a') as f:
        alltweets_string = alltweets.to_string(header=False,
                                               index = False)
        f.write(alltweets_string)
        
    return


#okay we need to clean the tweets now though so they dont have @ or emojis
def clean():
    tweets = '/Users/gavinkoma/Desktop/machinelearning/final_project/tweet.txt'
    with io.open(tweets,encoding='utf-8') as f:
        text = f.read().lower()
    print('corpus length:',len(text))
    
    #we should remove the @ parts as well but im not 100% sure how
    list = text.find_all('\n[\w* *]*: ', text)
        
    chars = sorted(list(set(text)))
    char_indices = dict((c,i) for i,c in enumerate(chars))
    indices_char = dict((i,c) for i,c in enumerate(chars))
    
    print('Unique Chars:', len(chars))
    return char_indices, indices_char

def main():
    #datainit()
    clean()

main()



