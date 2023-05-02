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
    tf.keras.layers.Dense(256, activation='relu', input_shape=(34,)),
    tf.keras.layers.Dense(128),
    tf.keras.layers.Dense(5)
    ])

    model.compile(optimizer='adam',
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

    return model

# Create a basic model instance
model = create_model()

# Display the model's architecture
model.summary()

checkpoint_path = "training_1/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

batch_size = 32

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, 
    verbose=1, 
    save_weights_only=True,
    save_freq=5*batch_size)

model.save_weights(checkpoint_path.format(epoch=0))


def train(train_data, train_labels, val_data, val_labels):
        
    # Train the model with the new callback
    model.fit(train_data, 
            train_labels,  
            epochs=10,
            validation_data=(val_data, val_labels),
            callbacks=[cp_callback])  # Pass callback to training

def evaluate(val_data, val_labels):
    latest = tf.train.latest_checkpoint(checkpoint_dir)
        # Load the previously saved weights
    model.load_weights(latest)

    # Re-evaluate the model
    loss, acc = model.evaluate(val_data, val_labels, verbose=2)
    print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

