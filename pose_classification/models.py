"""
Homework 5 - CNNs
CS1430 - Computer Vision
Brown University
"""

import tensorflow as tf
from keras.layers import \
       Conv2D, MaxPool2D, Dropout, Flatten, Dense

import hyperparameters as hp


class YourModel(tf.keras.Model):
    """ Your own neural network model. """

    def __init__(self):
        super(YourModel, self).__init__()
        self.optimizer = tf.keras.optimizers.experimental.SGD(learning_rate=hp.learning_rate, momentum=hp.momentum)

        self.architecture = [
            Dense(256, activation='relu'),
            Dense(128, activation='relu'),
            Dense(hp.num_classes, activation='softmax')
              ## Add layers here separated by commas.
        ]

    def call(self, x):
        """ Passes input image through the network. """

        for layer in self.architecture:
            x = layer(x)

        return x

    @staticmethod
    def loss_fn(labels, predictions):
        """ Loss function for the model. """
        
        loss_func = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        loss = loss_func(labels, predictions)

        return loss