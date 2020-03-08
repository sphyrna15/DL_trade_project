# DL_trade_project
First Project - Neural Networks for Time Series Prediction of financial assets

This is my first project on GitHub

I want to use Deep Learning methods for times series analysis. The goal is to find the best neural network architecture to predict
future prices of financial assets such as commodities or stocks.

I plan on comparing and experimenting with the following methods:
 - convolutional or dense neural networks (possibly in combination with RNNs)
 - recurrent neural network architectures such as LSTM or Attention networks
 - reinforcement learning methods (possibly also in combination with other machine learning methods)
 
 So far, I have completed the following steps:
 
 In the file Dataprep_class.py, I have created a class object that I use for preprocessing imported Data from sites like Quandl. The functionalities of this class include: 
 - Splitting the dataset into train, test and validation sets
 - Split data into sliding windows for training input, the label to each window is the timestep following the last elemement in the window
 - Scaling and rescaling the data to an intervall (0,1)
 - reshaping the data to fit the Keras recurent layers requirements
 
 In the file keras_LSTM1.py, I have created my first keras LSTM model for predicing daily Oil prices, trained on a BatchSize of 32 and in only 25 epochs, it achieves the following results: 
 

![First LSTM model tested](https://github.com/sphyrna15/DL_trade_project/blob/master/Model%20Evaluation/OPEC_LSTM1.png) 

I recently also tested how well a model can generalize between financial assets. The following figure shows two models: one that was trained on daily Oil prices for 50 epochs and one that was only trained on the same dataset for 1 epoch (control model). Both models were used to predict Gold prices (also daily):


![Transfer learning: trained on Oil, evaluated on Gold](https://github.com/sphyrna15/DL_trade_project/blob/master/Model%20Evaluation/Oil-Gold_LSTM.png)
