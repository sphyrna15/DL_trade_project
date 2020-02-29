# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 14:43:11 2020

Create training, validation and test sets with the imported numpy data

"""
import numpy as np

def sliding_windows(data, wsize, stepsize = 1):
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
    
    if 2 < data.ndim:
        raise ValueError("Incorrect number of dimensions: epected 2, received")
        print(data.ndim)
    
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


        
        
        
        
        
    
    




    
    
    
        
 
