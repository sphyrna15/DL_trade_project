# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 18:48:47 2020

Gold closing prices daily analysis

@author: timla
"""
from dataprep_class import Dataprep

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

data = data.to_numpy()
prices = np.delete(data, obj =  0, axis = 1)

#plot imported dataset

# plt.plot(prices[:,0])
# plt.xlabel("Date")
# plt.ylabel("Prices")

# plt.show()

prep = Dataprep()
scaled_data = prep.scaling(prices, "MinMax")
x, y = prep.sliding_windows(data = scaled_data, wsize = 60, stepsize = 1, rnn = True)

#x_train, y_train, x_test, y_test, x_val, y_val = prep.train_test_split(x, y, test_percent = 0.2, validation = True, val_percent = 0.1)
    

