# -*- coding: utf-8 -*-
"""
Create first LSTM prediction system in Keras

"""
from data_import import x_train , y_train, scaler
from Myplot_class import Myplot

from tensorflow import keras


model = keras.models.Sequential()

model.add(keras.layers.LSTM(units = 50, return_sequences = True, input_shape = (x_train.shape[1], 1)))
model.add(keras.layers.Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
model.add(keras.layers.LSTM(units = 50, return_sequences = True))
model.add(keras.layers.Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
model.add(keras.layers.LSTM(units = 50, return_sequences = True))
model.add(keras.layers.Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
model.add(keras.layers.LSTM(units = 50))
model.add(keras.layers.Dropout(0.2))

# Adding the output layer
model.add(keras.layers.Dense(units = 1))

model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['accuracy'])

model.fit(x_train, y_train, epochs = 25, batch_size = 32)
    





