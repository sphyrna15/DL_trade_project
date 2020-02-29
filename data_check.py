# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 18:48:47 2020

Gold closing prices daily analysis

@author: timla
"""
from split_data import sliding_windows

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


path = r"C:\Users\timla\Documents\Deep Learning Projects\Data"
filename = r"\OPEC-ORB.csv"

#Import dataset and drop irrelevant data
data = pd.read_csv(path + filename )

# data.drop('GBP (AM)' , axis = 1, inplace = True)
# data.drop('GBP (PM)' , axis = 1, inplace = True)
# data.drop('EURO (AM)' , axis = 1, inplace = True)
# data.drop('EURO (PM)' , axis = 1, inplace = True)
# data.drop('USD (PM)', axis = 1, inplace = True)

#Rearrange data to get plot in correct order

data = data.iloc[::-1]
data = data.reset_index()
data.drop('Date', axis = 1, inplace = True)

data2 = data.to_numpy()
prices = np.delete(data2, obj =  0, axis = 1)

#plot imported dataset

# plt.plot(prices[:,0])
# plt.xlabel("Date")
# plt.ylabel("Prices")

# plt.show()

x, y = sliding_windows(prices, wsize = 30, stepsize = 1)
    

