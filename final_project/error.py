#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 12:52:55 2022

@author: gavinkoma
"""

import matplotlib.pyplot as plt

epochs = [*range(1,6)]
loss1 = [0.7803, 0.5463, 0.4686, 0.4665, 0.4697]

loss2 = [0.8614, 0.4827, 0.4698, 0.7240, 0.8573]


plt.plot(epochs,loss2)

plt.plot([*range(1,6)], loss1)
plt.legend(['loss 128','loss 64'])
plt.xlabel('# of Epochs')
plt.ylabel('Computed Loss')
plt.title("Graph of Loss with Varied Batch Size")