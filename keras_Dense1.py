# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 17:21:33 2020

First DNN Model to try to approximate time series input of daily price data
"""

from data_import import x_train , y_train
from dataprep_class import Dataprep

from tensorflow import keras

model = keras.models.Sequential()

model.add(keras.layers.BatchNormalization(input_shape = [x_train.shape[1]]))
model.add(keras.layers.Dense(300, activation = 'elu'))
model.add(keras.layers.Dense(200, activation = 'elu'))
model.add(keras.layers.Dense(100, activation = 'elu'))
model.add(keras.layers.Dense(1, activation = 'relu'))

optimizer = keras.optimizers.Adam(lr = 0.0075, amsgrad = True)
model.compile(loss = 'mean_squared_error', optimizer = optimizer, metrics = ['accuracy'])


# make sure data is correct format
prep = Dataprep()
x_train = prep.rnn_reshape(data = x_train, to_rnn = False)



history = model.fit(x_train, y_train, epochs = 25, batch_size = 32)