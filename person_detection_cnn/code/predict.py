from .preprocess import Datasets
from .models import YourModel
import tensorflow as tf
import os
from skimage.io import imread
from skimage.transform import resize
import numpy as np
from .hyperparameters import img_size
from typing import Union

def predict_label(img: Union[str, np.ndarray], model: any, datasets: Datasets):
    '''Returns the label of a prediction for a image given a model and dataset.'''

    # processing image
    if isinstance(img, str):
        image = imread(img)
    else:
        image = img

    if len(image.shape) == 2:
        image = np.stack([image, image, image], axis=-1)
    image = resize(image, (img_size, img_size, 3), preserve_range=True)
    image = datasets.preprocess_fn(image)
    image = np.expand_dims(image, 0)

    # predicting
    prediction = model.predict(image)
    # print(prediction)
    # print(np.argmax(prediction[0]))
    # print(datasets.idx_to_class)
    label = datasets.idx_to_class[np.argmax(prediction[0])]

    return label

def determine_person(imgpath):
    '''Returns whether the image is a person or not.'''
    # loading model
    # path_to_weights = "./checkpoints/your_model/043023-161852/your.weights.e002-acc0.9718.h5"
    path_to_weights = "./checkpoints/your_model/050423-180546/your.weights.e002-acc0.8320.h5"

    model = YourModel()
    model(tf.keras.Input(shape=(224, 224, 3)))
    model.load_weights(path_to_weights)

    datasets = Datasets('..'+os.sep+'data'+os.sep, "TASK 1")

    # processing image
    image = imread(imgpath)
    if len(image.shape) == 2:
        image = np.stack([image, image, image], axis=-1)
    image = resize(image, (img_size, img_size, 3), preserve_range=True)
    image = datasets.preprocess_fn(image)
    image = np.expand_dims(image, 0)

    # predicting
    prediction = model.predict(image)
    label = datasets.idx_to_class[np.argmax(prediction[0])]

    print(label)
    if label == "0":
        return False
    return True

# determine_person("../data/test_inputs/no-human-2.jpg")