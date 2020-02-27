# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 18:48:47 2020

Gold closing prices daily analysis

@author: timla
"""

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
# data = data.drop('Date', axis = 1, inplace = True)


#Rearrange data to get plot in correct order

data = data.iloc[::-1]
data = data.reset_index()
data.drop('index', axis = 1, inplace = True)


prices = data.to_numpy()

#plot imported dataset

plt.plot(prices[:,0])
plt.xlabel("Date")
plt.ylabel("Prices")

plt.show()


# def sliding_windows(data, window_size):
    

