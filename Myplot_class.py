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
    
    def compare_models(self, real_price, model1_preds, model2_preds, title, name1 = "Model 1", name2 = "Model 2"):
        """
        Parameters
        ----------
        real_prices : numpy array
            Real prices of the asset
        model1_preds : numpy array
            Prices predicted by the first model
        model2_preds : numpy array
            Prices predicted by the second model
        title : string
            Plot title (fincancial asset name)
        name1 / name2 : string
            Model 1 and 2 names
        -------
        returns: Nothing, just plots figure
        """

        plt.plot(real_price, color = 'green', label = "Real Prices")
        plt.plot(model1_preds, color = 'red', label = name1)
        plt.plot(model2_preds, color = 'blue', label = name2)
        plt.title(title)
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        plt.grid()
        plt.show()

        return None