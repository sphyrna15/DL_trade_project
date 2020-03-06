# DL_trade_project
First Project - Neural Networks for Time Series Prediction of financial assets

This is my very first project on GitHub

I want to use Deep Learning methods for times series analysis. The goal is to find the best neural network architecture to predict
future prices of financial assets and investment instruments.

I plan on comparing the following methods:
 - Convolutional NN (in combi with RNNs)
 - more sophisticated RNN architectures such as LSTM or Attention NN
 - reinforcement learning
 
 So far, I have completed the following steps:
 
 In the file Dataprep_class.py, I have created a class object that I use for preprocessing imported Data from sites like Quandl. The functionalities of this class include: 
 - Splitting the dataset into train, test and validation sets
 - Split data into sliding windows for training input, the label to each window is the timestep following the last elemement in the window
 - Scaling and rescaling the data to an intervall (0,1)
 - reshaping the data to fit the Keras recurent layers requirements
 
 In the file keras_LSTM1.py, I have created my first keras LSTM model for predicing financial asset prices, trained on a BatchSize of 32 and in only 25 epochs, it achieves the following results: 
 

![First LSTM model tested](https://github.com/sphyrna15/DL_trade_project/blob/master/OPEC_LSTM1.png) 
