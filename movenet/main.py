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

def image_prediction(image):
  person = detect(image)
  embeddings = utils.get_embedding(person)
  #print(embeddings)
  # print(person)
  img = utils.draw_prediction_on_image(image, person, crop_region=None, 
                              close_figure=False, keep_input_size=True)
  return img

def live_camera_loop(): 
  # define a video capture object
  vid = cv2.VideoCapture(0)
    
  while(True):
        
      # Capture the video frame
      # by frame
      ret, frame = vid.read()

      results = image_prediction(frame)
    
      # Display the resulting frame
      cv2.imshow('frame', results)
        
      # the 'q' button is set as the
      # quitting button you may use any
      # desired button of your choice
      if cv2.waitKey(1) & 0xFF == ord('q'):
          break
    
  # After the loop release the cap object
  vid.release()
  # Destroy all the windows
  cv2.destroyAllWindows()

def main():
  # test_image_url = "../data/train/tree/00000125.jpg"
  # test_image = cv2.imread(test_image_url)
  # results = image_prediction(test_image)
  # cv2.imshow("window", results)
  # cv2.waitKey(0)
  live_camera_loop()
    
main()