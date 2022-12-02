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

for i in range(0, len(cleantweets_string) - maxlen, step):
    sentences.append(cleantweets_string[i: i + maxlen])
    next_chars.append(cleantweets_string[i + maxlen])
print('nb sequences:', len(sentences))

print('Vectorization...')
x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1
print('Done!')

# build the model: a single LSTM
print('Build model...')
model = Sequential()
model.add(LSTM(128, input_shape=(maxlen, len(chars))))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))

optimizer = RMSprop(learning_rate=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)
print('Done!')

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.ma.log(preds)
    preds = preds.filled(0)
    preds = preds / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def on_epoch_end(epoch, logs):
    # clear_output()
    # Function invoked at end of each epoch. Prints generated text.
    print()
    print('----- Generating text after Epoch: %d' % (epoch+1))
    
    start_index = random.randint(0, len(cleantweets_string) - maxlen - 1)
    text_shuffled = cleantweets_string.split(",")
    random.shuffle(text_shuffled)
    text_shuffled = ','.join(map(str,text_shuffled))
    start_index = random.randint(0, len(text_shuffled) - maxlen - 1)
    
    for diversity in [0.2, 0.4, 0.5, 1.0]:
        print('----- diversity:', diversity)

        generated = ''
        sentence = cleantweets_string[start_index: start_index + maxlen]
        generated += sentence
        print('----- Generating with seed: "' + sentence + '"')
        sys.stdout.write(generated)

        for i in range(400):
            x_pred = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x_pred[0, t, char_indices[char]] = 1.

            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()

print_callback = LambdaCallback(on_epoch_end=on_epoch_end)

epoch_val = [*range(1,10)]

model.fit(x, y,
          batch_size=128,
          epochs=1,
          callbacks=[print_callback])







