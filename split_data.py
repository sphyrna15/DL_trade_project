# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 14:43:11 2020

Create training, validation and test sets with the imported numpy data

"""
import numpy as np


class Dataprep():
    
    """
    Data Preparation functions for Time Series Data
    from Quandl.com or similar platforms"""
    
    
    def __init__(self, input_shape):
        """ Initialize useful default object variables """
        
        self.input_shape = input_shape
        self.steps = 1
        self.test_percent = 0.1
        
    
    def sliding_windows(self, data, wsize, stepsize = None):
        """
        Parameters
        ----------
        data : numpy array
            contains data to be sliced into sets
        wsize : integer
            window size for input data of network
        stepsize: integer
            window sliding distance
            
        test and training examples that are sliced into wsize long windows as input
        labeled with the next time step after window ends
        -------
        returns: sliced dataset for training, labels
        """
        if stepsize == None:
            stepsize = self.stepsize
        
        if 2 < data.ndim:
            raise ValueError("Incorrect number of dimensions: expected 2")
        
        if stepsize < 1:
            raise ValueError("stepsize cannot be zero or negative")
            
        if wsize > data.shape[0]:
            raise ValueError("Window size cannot exceed  data array size")
        
        length = data.shape[0]                         # compute length of data sequence
        steprest = (length - wsize) % stepsize         # find out correction for window size
        num_steps = int((length - wsize) / stepsize)   # find how many steps can be taken with stepsize
        num_steps -= 1   #needs to be adjusted for indexing in loop
        
        num_examples = num_steps + steprest
        
        sliced_data = np.zeros((wsize, num_examples))  # initialize examples and 
        labels = np.zeros((1, num_examples))           # labels array with zeros
        
        
        for i in range(0, num_steps, stepsize):
        
            sliced_data[:, i] = data[i : i + wsize, 0]
            labels[0, i] = data[i + wsize, 0]
        
        for j in range(num_steps, num_examples, 1):
        
            sliced_data[:, j] = data[j : j + wsize, 0]
            labels[0, j] = data[j + wsize+1, 0]
            
        return sliced_data, labels
    
    
    def train_test_split(self, data, labels, test_percent = None, validation = False, val_percent = None):
        """
        Parameters
        ----------
        data : numpy array
            contains data to be split into train, test (and validation) sets
        labels : numpy array
            corresponding labels for data
        test_percent : integer between zero and one
            percentage of dataset to be used as test set
        validation : bool
            False - do not output a validation set
        val_percent : integer between zero and one
            percentage of dataset to be used as validation set
            
        Returns
        -------
        x_train, y_train, x_test, y_test (optional: x_val, y_val) 
        """
        if test_percent == None:
            test_percent = self.test_percent
            
        if validation == False and val_percent != None:
            raise ValueError("Recieved unexpected value for val_percent - no input for validation percentage required")
            
        if labels.ndim != data.ndim:
            raise ValueError("Dimensions of data and labels dot not agree")
            
        if test_percent <= 0 or test_percent >= 1:
            raise ValueError("test_percent is out of range, should be between 0 and 1")
            
        
        
        
        return None