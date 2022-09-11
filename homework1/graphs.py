# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# read this csv file, remember to put the full path to 
# the directory where you saved the data
df = pd.read_csv('cars.csv')  # df is DataFrame object
#print (df.head(10))    # see the first 5 rows of the loaded tableprint (list(df.columns))

print("Print all cars that have more than 3 cylinders:")
print (df[df['Model']==70])

df1970 = df[df['Model']==70]
