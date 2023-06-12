#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 19:56:55 2022

@author: gavinkoma
"""

from __future__ import print_function
from keras.callbacks import LambdaCallback
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from tensorflow.keras.optimizers import RMSprop
import numpy as np
import random
import re
import sys
import pandas as pd

#okay start by downloading data and reading through it
start = pd.read_csv('TweetsElonMusk2010_2021.csv')
new = pd.read_csv('rawdata2022.csv')

tweets2010_2021 = start['tweet']
tweets2022 = new['Tweets']

#append the data to one list
alltweets = pd.DataFrame(tweets2010_2021.append(tweets2022, 
                                                ignore_index = True))

alltweets.columns = ['uncleaned']

alltweets.astype(str)

def cleantwt(twt):
    emoj = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642" 
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
            "]+", re.UNICODE)
    
    twt = re.sub('RT', '', twt) # remove 'RT' from tweets
    twt = re.sub('#[A-Za-z0-9]+', '', twt) # remove the '#' from the tweets
    twt = re.sub('\\n', '', twt) # remove the '\n' character
    twt = re.sub('https?:\/\/\S+', '', twt) # remove the hyperlinks
    twt = re.sub('@[\S]*', '', twt) # remove @mentions
    twt = re.sub('^[\s]+|[\s]+$', '', twt) # remove leading and trailing whitespaces
    twt = re.sub(emoj, '', twt) # remove emojis
    return twt

alltweets['cleaned_tweets'] = alltweets['uncleaned'].apply(cleantwt)

#but also some tweets were just emoji and retweets so we need to remove the blanks
alltweets.drop(alltweets[alltweets['cleaned_tweets'] == ''].index,inplace = True)

cleantweets = alltweets['cleaned_tweets']


#lets make it a text file? 
with open('tweet.txt','w') as f:
    pd.options.display.max_colwidth = 300 #character limit of twitter
    cleantweets_string = cleantweets.to_string(header=None,
                                            index = None)
    f.write(cleantweets_string)

#okay so now we should start modeling the RNN
#we want to look at the values in cleentweets string

chars = sorted(list(set(cleantweets_string)))
print('total chars:', len(chars)) #191 chars
char_indices = dict((c,i) for i,c in enumerate(chars))
indices_char = dict((i,c) for i,c in enumerate(chars))


maxlen = 100
step = 5
sentences = []
next_chars = []


text_shuffled = cleantweets_string.split("\n")
random.shuffle(text_shuffled)
text_shuffled = ','.join(map(str,text_shuffled))
start_index = random.randint(0, len(text_shuffled - maxlen - 1))

