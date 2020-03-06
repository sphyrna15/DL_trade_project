# -*- coding: utf-8 -*-

"""
Class for plotting and evaluating various models and their predictions

"""

import numpy as np
import matplotlib.pyplot as plt

class Myplot():

    """ Plot predictions to evaluate different models """

    def real_vs_pred(self, real_prices, predicted_prices, title):
        """
        Parameters
        ----------
        real_prices : numpy array
            Real prices of the asset
        predicted_prices : numpy array
            Prices predicted by the model
        title : string
            Plot title (fincancial asset name)
        -------
        returns: Nothing, just plots figure
        """

        plt.plot(real_prices, color = 'red', label = 'Real prices')
        plt.plot(predicted_prices, color = 'blue', label = 'Predicted prices')
        plt.title(title)
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        plt.grid()
        plt.show()
        
        return None