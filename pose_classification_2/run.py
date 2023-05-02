import os
import sys
import argparse
import re
from datetime import datetime
import tensorflow as tf

from skimage.transform import resize

from skimage.io import imread
from lime import lime_image
from skimage.segmentation import mark_boundaries
from matplotlib import pyplot as plt
import numpy as np

def create_model():
    model = tf.keras.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
    ])

    return model

def train(model, train_data, train_labels, val_data, val_labels):
        
    model.compile(optimizer='adam', 
                loss=tf.losses.CategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

    history = model.fit(train_data, train_labels, epochs=10, 
                    validation_data=(val_data, val_labels))



