# -*- coding: utf-8 -*-
"""
Goal: Advance keras knowledge - get into hyperparameters, tensorboard and clean up code

"""


import tensorflow as tf
from tensorflow import keras

from keras_LSTM2 import lstm
from data_import import x_train, y_train, x_val, y_val

timesteps = x_train.shape[1]

lstm1 = lstm()

# build model 1 with 3 layer and 50 units, dropout 0.2, not L2 reg
model1 = lstm1.build(3, 50, timesteps)

tb_callback = keras.callbacks.TensorBoard(histogram_freq = 1)

model1.fit(x_train, y_train, epochs = 15, validation_data = (x_val, y_val), callbacks = [tb_callback])






