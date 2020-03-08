# -*- coding: utf-8 -*-
"""
Goal: Advance keras knowledge - get into hyperparameters, tensorboard and clean up code

"""
import tensorflow as tf
from tensorflow import keras
import numpy as np

class lstm():
    """ class to clean up keras lstm structure and advance training """

    def build(self, layers, units, timesteps, loss = 'mean_squared_error', dropout = True, l2_reg = False, dropout_rate = 0.2):
        """
        Parameters
        ----------
        layers : integer - how many LSTM layers to use
        units : integer - how many units per LSTM layer
        timesteps : x_train.shape[1] (time steps per input frame)
        loss : loss function
        dropout : bool - use dropout or not
        l2_reg : bool - use L2 regularization or not
        dropout_rate : float between zero and one
        -------
        returns: compiled LSTM model
        """
        if layers <= 0 or units <= 0:
            raise ValueError("layers and units cannot be smaller or equal to zero")
        
        if 0 > dropout_rate or dropout_rate > 1:
            raise ValueError("dropout rate must be between zero and 1")

        if l2_reg:
            reg = keras.regularizers.l2(0.001)
        else:
            reg = None

        if not dropout:
            dropout_rate = 0
        

        model = keras.models.Sequential()

        if layers == 1:
            model.add(keras.layers.LSTM(units = 50, return_sequences = False, kernel_regularizer = reg, input_shape = (timesteps, 1))) 
            model.add(keras.layers.Dropout(dropout_rate))
        

        else:
            model.add(keras.layers.LSTM(units = 50, return_sequences = True, kernel_regularizer = reg, input_shape = (timesteps, 1))) 
            model.add(keras.layers.Dropout(dropout_rate))

            for i in range(layers - 2):
                model.add(keras.layers.LSTM(units = 50, return_sequences = True, kernel_regularizer = reg)) 
                model.add(keras.layers.Dropout(dropout_rate))
            
            model.add(keras.layers.LSTM(units = 50, return_sequences = False, kernel_regularizer = reg)) 
            model.add(keras.layers.Dropout(dropout_rate))

        model.add(keras.layers.Dense(1, activation = 'relu'))

        optimizer = keras.optimizers.Adam(learning_rate = 0.0075, epsilon = 1e-08)
        model.compile(optimizer = optimizer, loss = loss, metrics = ['accuracy'])

        return model

            






