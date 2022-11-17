#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 10:45:02 2022

@author: gavinkoma
"""
#import libraries
import pandas as pd
import numpy as np

#okay start by downloading data and reading through it
start = pd.read_csv('TweetsElonMusk2010_2021.csv')
new = pd.read_csv('rawdata2022.csv')

tweets2010_2021 = start['tweet']
tweets2022 = new['Tweets']

#append the data to one list
alltweets = pd.DataFrame(tweets2010_2021.append(tweets2022, 
                                                ignore_index = True))

#we also should concatenate the list into one big list instead of a list
#with a variety of indices 
'' = .join(alltweets)



#okay we need to clean the tweets now though so they dont have @ or emojis
def removeunwantedtext():
    
    
    
    
    return




