# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 15:50:04 2020

Deep Learning and Tensorboard refresher

@author: timla
"""

import tensorflow as tf
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt
import os

"""root_logdir = os.path.join(os.curdir, 'my_logs') #for tensorboard log setup
def get_run_logdir():
    import time
    run_id = time.strftime("run_%Y_%m_%d_-%H_%M_%S")
    return os.path.join(root_logdir, run_id) """


mnist = keras.datasets.fashion_mnist
(x_train, y_train) , (x_test, y_test) = mnist.load_data()

classes = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot"]



model = keras.models.Sequential()

model.add(keras.layers.Flatten(input_shape = [28, 28]))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dense(300, activation = 'elu'))
model.add(keras.layers.Dense(200, activation = 'elu'))
model.add(keras.layers.Dense(100, activation = 'elu'))
model.add(keras.layers.Dense(10, activation = 'softmax'))

optimizer = keras.optimizers.Adam(lr = 0.0075, amsgrad = True)
model.compile(loss = 'sparse_categorical_crossentropy', optimizer = optimizer, metrics = ['accuracy'])

""" run_logdir = get_run_logdir() 
tensorboard_log = keras.callbacks.TensorBoard(run_logdir) #Tensorboard log setup """


history = model.fit(x_train, y_train, epochs = 10, validation_data = (x_test, y_test))

pd.DataFrame(history.history).plot(figsize = (8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1) #sets vertical range between 0 and 1
plt.show()

