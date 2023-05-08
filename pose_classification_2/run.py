import os
import sys
import argparse
import re
from datetime import datetime
import tensorflow as tf
import pandas as pd

from skimage.transform import resize
from skimage.io import imread
from lime import lime_image
from skimage.segmentation import mark_boundaries
from matplotlib import pyplot as plt
from movenet.movenet import Movenet
from movenet.utils import get_embedding
import numpy as np
import cv2
from constants import num_to_label, label_to_num

def detect(input_tensor, inference_count=3):
    """Runs detection on an input image.

    Args:
        input_tensor: A [height, width, 3] Tensor of type tf.float32.
        Note that height and width can be anything since the image will be
        immediately resized according to the needs of the model within this
        function.
        inference_count: Number of times the model should run repeatly on the
        same input image to improve detection accuracy.

    Returns:
        A Person entity detected by the MoveNet.SinglePose.
    """
    image_height, image_width, channel = input_tensor.shape

    movenet = Movenet('movenet/lite-model_movenet_singlepose_thunder_tflite_float16_4')

    # Detect pose using the full input image
    movenet.detect(input_tensor, reset_crop_region=True)

    # Repeatedly using previous detection result to identify the region of
    # interest and only croping that region to improve detection accuracy
    for _ in range(inference_count - 1):
        person = movenet.detect(input_tensor, 
                                reset_crop_region=False)

    return person

def create_model():
    num_classes = 9
    model = tf.keras.Sequential([
    tf.keras.layers.Dense(256, activation='relu', input_shape=(34,)),
    tf.keras.layers.Dense(128),
    tf.keras.layers.Dense(num_classes)
    ])

    model.compile(optimizer='adam',
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

    return model

# Create a basic model instance
model = create_model()

# Display the model's architecture
model.summary()

# checkpoint_path = "./training_1/cp-{epoch:04d}.ckpt"
# checkpoint_dir = os.path.dirname(checkpoint_path)

# batch_size = 32

# # Create a callback that saves the model's weights
# cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, 
#     verbose=1, 
#     save_weights_only=True,
#     save_freq=5*batch_size)

# model.save_weights(checkpoint_path.format(epoch=0))

checkpoint_filepath = "best_weights_testing.h5"
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_sparse_categorical_accuracy',
    mode='max',
    save_best_only=True)
csv_callback = tf.keras.callbacks.CSVLogger('training_accuracies.csv', append=True)

def train(train_data, train_labels, val_data, val_labels):
        
    # Train the model with the new callback
    model.fit(train_data, 
            train_labels,  
            epochs=50,
            batch_size=32,
            validation_data=(val_data, val_labels),
            callbacks=[cp_callback, csv_callback])  # Pass callback to training

def model_train():
    train_data_labels = pd.read_csv("../csv_data/train_data_updated.csv").to_numpy()
    test_data_labels = pd.read_csv("../csv_data/test_data_updated.csv").to_numpy()
    train_data = train_data_labels[:, :34]
    train_labels = train_data_labels[: ,34]
    test_data = test_data_labels[:, :34]
    test_labels = test_data_labels[: ,34]
    train(train_data, train_labels, test_data, test_labels)

def get_prediction(data):
    model.load_weights(checkpoint_filepath)
    prediction = model.predict(data)
    return prediction

def evaluate(val_data, val_labels):
    model.load_weights(checkpoint_filepath)

    # Re-evaluate the model
    loss, acc = model.evaluate(val_data, val_labels, verbose=2)
    print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

# model.load_weights(checkpoint_filepath)

# def evaluate(val_data, val_labels):
#     latest = tf.train.latest_checkpoint(checkpoint_dir)
#         # Load the previously saved weights
#     model.load_weights(latest)

#     # Re-evaluate the model
#     loss, acc = model.evaluate(val_data, val_labels, verbose=2)
#     print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

if __name__ == "__main__":
    model_train()
    # test_img_path = "../testing_images/00000016.jpg"
    # img = cv2.imread(test_img_path)
    # person = detect(img)
    # embedding = np.expand_dims(get_embedding(person), 0)
    # pred = get_prediction(embedding)
    # label = num_to_label[np.argmax(pred[0])]
    # print(label)