# -*- coding: utf-8 -*-
"""
Create first LSTM prediction system in Keras

"""
from data_import import x,y

from tensorflow import keras


class LSTM_model1():
    """ First LSTM model implementation and training tryout() """

    def __init__():

        self.activation = 'tanh'
        self.dropout_rate = 0.2

    def build(x_shape, units, layers, activation = None, dropout = True, dropout_rate = None):
        """
        Parameters
        ----------
        x_shape : Vetctor
            Input shape to be passed to network
        units : integer
            how many units to use in LSTM layer
        layers : integer >= 2
            how many LSTM layers to build
        activation : String
            Activation function - default is tanh
        dropout : bool
            Include Dopout layer after every LSTM layer
        dropout_rate : float
            Rate to implement Dropout
        -------
        returns: compiled keras model
        """

        if units == 0 or layers == 0:
            raise ValueError("Units and layers cannot be zero")

        if dropout_rate == None and dropout:
            dropout_rate = self.dropout_rate
        
        if activation == None:
            activation = self.activation
        
        if 0 > dropout_rate or dropout_rate >= 1:
            raise ValueError("Dropout rate out of bounds")

        model = keras.models.Sequential()

        # add input layer
        model.add(keras.layers.LSTM(units = units, activation = activation, return_sequences = True, input_shape = x_shape))

        # adding speciefied number of layers 
        for i in range(layers - 2):

            model.add(keras.layers.LSTM(units = units, activation = activation, return_sequences = True))

            if dropout:
                model.add(keras.layers.Dropout(dropout_rate))
        
        # Adding a last LSTM layer and some Dropout regularisation
        model.add(keras.layers.LSTM(units = 50, activation = activation))
        model.add(keras.layers.Dropout(0.2))

        # Adding the output layer
        model.add(keras.layers.Dense(units = 1 activation = 'relu'))

        # Compile model with Adam optimization and MSE loss
        model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['accuracy'])

        return model




