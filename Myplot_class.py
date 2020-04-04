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

    def pred_examples(self, x_test, real_prices, predictions, indices):
        """
        Parameters
        ----------
        x_test : numpy array
            tested data for visualization
        real_prices : numpy array
            Real prices of the examples
        predicted_prices : numpy array
            Prediction examples
        indices : python list
            list of indices to plot
                
        Attention :
        All points in the price arrays will be plotted in subplots - don't use too many
        make sure that the arrays are of equal length
        -------
        returns: Nothing, just plots figure
        """

        length = len(indices)
        wsize = x_test.shape[1]
        x_show = x_test[:, wsize - 21 : ]

        if length != 6:
            raise ValueError("Only 6 examples can be plotted at once")

        figure, axes = plt.subplots(2, 3)
        k = 0
        t = 0

        for i in range(6):
            if i > 2:
                k = 1
                i = i%3
            j = indices[t]; t += 1
            
            axes[k,i].plot(x_show[j, :], label = 'Price Histroy')
            axes[k,i].plot(21, real_prices[j], 'go', label = 'Future Price')
            axes[k,i].plot(21, predictions[j], 'r+', label = 'Predicted Price')
            axes[k,i].set_title(label = str(j))
            axes[k,i].set_xlabel('Time')
            axes[k,i].set_ylabel('Price')
            axes[k,i].legend()
            axes[k,i].grid()
        plt.show()
        
        return None


