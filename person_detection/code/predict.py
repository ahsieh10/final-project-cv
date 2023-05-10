from .preprocess import Datasets
import tensorflow as tf
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
    label = datasets.idx_to_class[np.argmax(prediction[0])]

    return label