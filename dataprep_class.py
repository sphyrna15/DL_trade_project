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
    
    
    def __init__(self):
        """ Initialize useful default object variables """
        
        self.steps = 1
        self.train_percent = 0.9
        self.scale_types = ("MinMax", "Standard")
        
    
    def sliding_windows(self, data, wsize, stepsize = None):
        """
        Parameters
        ----------
        data : numpy array
            contains data to be sliced into sets
        wsize : integer
            window size for input data of network
        stepsize : integer
            window sliding distance
        rnn : bool
            data to be reshaped for rnn/lstm -> wsize will be num_timesteps
            
        test and training examples that are sliced into wsize long windows as input
        labeled with the next time step after window ends
        -------
        returns: sliced dataset for training, labels
        """
        if stepsize == None:
            stepsize = self.steps
        
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
        
        sliced_data = np.transpose(sliced_data)
        labels = np.transpose(labels)
                    
        return sliced_data, labels
    
    
    def train_test_split(self, data, labels ,shuffle = False, train_percent = None, validation = False, val_percent = None):
        """
        Parameters
        ----------
        data : numpy array
            contains data to be split into train, test (and validation) sets
        labels : numpy array
            corresponding labels for data
        shuffle : bool
            shuffle data or not
        train_percent : integer between zero and one
            percentage of dataset to be used as train set
        validation : bool
            False - do not output a validation set
        val_percent : integer between zero and one
            percentage of dataset to be used as validation set
            
        Returns
        -------
        x_train, y_train, x_test, y_test (optional: x_val, y_val) 
        """
        if train_percent == None:
            train_percent = self.train_percent
            
        if validation == False and val_percent != None:
            raise ValueError("Recieved unexpected value for val_percent - no input for validation percentage required")

        if validation == True and val_percent == None:
            raise ValueError("Missing one required argument: val_percent")  

        if labels.ndim != data.ndim:
            raise ValueError("Dimensions of data and labels dot not agree")
            
        if train_percent <= 0 or train_percent >= 1:
            raise ValueError("train_percent is out of range, should be between 0 and 1")
            
        num_examples = data.shape[0] 
        num_train_examples = int(num_examples * train_percent)    # get number of train examples for indexing

        # shuffle data and labels in unison
        if shuffle:
            rng_state = np.random.get_state()
            np.random.shuffle(data)
            np.random.set_state(rng_state)
            np.random.shuffle(labels)

        # now, reindex the datasets into train and train sets
        if not validation:
            x_train = data[ : num_train_examples, : ]
            y_train = labels[ : num_train_examples, :]
            
            x_test = data[num_train_examples : , : ]
            y_test = labels[num_train_examples : , : ]

            return x_train, y_train, x_test, y_test

        # If a validation set is wanted, reindex accordingly
        num_val_examples = int(num_examples * val_percent)
        val_index = num_train_examples + num_val_examples

        x_train = data[ : num_train_examples, : ]
        y_train = labels[ : num_train_examples, :]

        x_val = data[num_train_examples : val_index, : ]
        y_val = labels[num_train_examples : val_index, : ]

        x_test = data[val_index : , : ]
        y_test = labels[val_index : , : ]

                
        return x_train, y_train, x_test, y_test, x_val, y_val

    def scaling(self, data, scale_type):
        """
        Parameters
        ----------
        data : numpy array
            contains data to be scaled
        scale_type : string
            type of scaling to be applied (MinMax, ...)
                    
        Returns
        -------
        scaled data, scaler
        """

        if scale_type not in self.scale_types:
            raise ValueError("Not a valid or recognized scaling method")

        if scale_type == "MinMax": # MinMax scaling on interval (0,1)
            from sklearn.preprocessing import MinMaxScaler

            scaler = MinMaxScaler(feature_range = (0,1))
            scaled_data = scaler.fit_transform(data)
            return scaled_data, scaler

        if scale_type == "Standard":   # Standard scaling
            from sklearn.preprocessing import StandardScaler
            
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(data)
            return scaled_data, scaler

    def inverse_scaling(self, data, scaler):
        r"""
        Parameters
        ----------
        data : numpy array
            contains data to be reverse scaled
        scaler : sklearn.preprocessing object
            scale object used to scale the data, returned by Dataprep.scaling()
                    
        Returns
        -------
        inverse scaled data 
        """

        return scaler.inverse_transform(data)

    def rnn_reshape(self, data, to_rnn = True):
        """ Reshape data for keras recurrent network layers
        Parameter : data (numpy array) to be reshaped 
        to_rnn (bool) : True -> reshape to RNN shape, Flase -> reshape back to normal from RNN shape """

        if to_rnn:
            return np.reshape(data, (data.shape[0], data.shape[1], 1))

        return np.reshape(data, (data.shape[0], data.shape[1]))
    
    def pd_to_np(self, data, colidx):
        """ handle QUANDL pandas dataframes and return clean 1-dim numpy array 
        Parameters: 
            data : pd dataframe
            colidx : collum index of dataframe """
        
        if colidx > data.shape[1]:
            raise ValueError("Index out of bounds, found shape " + str(data.shape))
        
        data = data.iloc[::-1]
        data = data.reset_index()
        data.drop('Date', axis = 1, inplace = True)
        colidx -= 1
        
        data = data.to_numpy()
        data = data[:, colidx]
        
        return data.reshape((-1,1))
    
    def concat(self, data1, data2):
        """ Concatenate two arrays of dimension (None, 1) 
        Parameters:
            data1 and data2 : numpy arrays """
        
        data = np.concatenate((data1, data2), axis = 0)
        
        return data.reshape((-1,1)) 
            
        
        
        
        
        

        

        
