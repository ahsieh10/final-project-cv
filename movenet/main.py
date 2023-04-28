import csv
import cv2
import itertools
import numpy as np
import pandas as pd
import os
import sys
import tempfile
import tqdm

from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import utils
from data import BodyPart
from movenet import Movenet
movenet = Movenet('lite-model_movenet_singlepose_thunder_tflite_float16_4')

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

  # Detect pose using the full input image
  movenet.detect(input_tensor, reset_crop_region=True)

  # Repeatedly using previous detection result to identify the region of
  # interest and only croping that region to improve detection accuracy
  for _ in range(inference_count - 1):
    person = movenet.detect(input_tensor, 
                            reset_crop_region=False)

  return person


def main():
  test_image_url = "../data/train/tree/00000125.jpg"
  if len(test_image_url):
    image = cv2.imread(test_image_url)
    person = detect(image)
    embeddings = utils.get_embedding(person)
    #print(embeddings)
    # print(person)
    img = utils.draw_prediction_on_image(image, person, crop_region=None, 
                                close_figure=False, keep_input_size=True)
    cv2.imshow("window",img)
    cv2.waitKey(0)
    
main()