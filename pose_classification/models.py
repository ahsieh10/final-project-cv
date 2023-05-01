"""
For documentation of different layers, please refer to torch.nn
https://pytorch.org/docs/stable/nn.html
"""

import tensorflow as tf
from keras.layers import \
       Conv2D, MaxPool2D, Dropout, Flatten, Dense, Activation

import hyperparameters as hp


class YourModel(tf.keras.Model):
    """ Your own neural network model. """

    def __init__(self):
        super(YourModel, self).__init__()
        
        self.optimizer = tf.keras.optimizers.Adam()

        self.architecture = [
              ## Add layers here separated by commas.
            Dense(256, activation='relu'),
            Dense(128, activation='relu'),
            Dense(5, activation='softmax')
          
        ]

    def call(self, x):
        """ Passes input image through the network. """

        for layer in self.architecture:
            x = layer(x)

        return x

    @staticmethod
    def loss_fn(labels, predictions):
        """ Loss function for the model. """

        return tf.keras.losses.SparseCategoricalCrossentropy()(labels, predictions)



