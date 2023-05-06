'''Entrypoint for our application'''

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import os
import sys

import movenet.utils as utils
from movenet.data import BodyPart
from movenet.movenet import Movenet
from person_detection_cnn.code.models import YourModel as PDModel
from person_detection_cnn.code.models import VGGModel as PDVGG
from person_detection_cnn.code.preprocess import Datasets as PDDatasets
from person_detection_cnn.code.predict import predict_label
from constants import num_to_label, label_to_num


movenet = Movenet('movenet/lite-model_movenet_singlepose_thunder_tflite_float16_4')

'''Load human detection model'''
person_detect_weights_path = "./person_detection_cnn/code/checkpoints/vgg/vgg.weights.e012-acc0.9692.h5"
person_detect_base_path = "./person_detection_cnn/code/vgg16_imagenet.h5"
# person_detect_weights_path = "./person_detection_cnn/code/checkpoints/your_model/your.weights.e018-acc0.8703.h5"

person_detect_model = PDVGG()
# person_detect_model = PDModel()

person_detect_model(tf.keras.Input(shape=(224, 224, 3)))

person_detect_model.vgg16.load_weights(person_detect_base_path, by_name=True)
person_detect_model.head.load_weights(person_detect_weights_path, by_name=False)
# person_detect_model.load_weights(person_detect_weights_path)

person_detect_datasets = PDDatasets('./person_detection_cnn'+os.sep+'data'+os.sep, "TASK 1")

'''Load pose classifier model'''

pose_weights_path = "./pose_classification_2/best_weights_updated.h5"

def create_model():
    model = tf.keras.Sequential([
    tf.keras.layers.Dense(256, activation='relu', input_shape=(34,)),
    tf.keras.layers.Dense(128),
    tf.keras.layers.Dense(9)
    ])

    model.compile(optimizer='adam',
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

    return model

# Create a basic model instance
pose_model = create_model()
pose_model.load_weights(pose_weights_path)

def predict_pose(embedding):
    embedding = np.expand_dims(embedding, 0)
    pred = pose_model.predict(embedding)
    max = np.max(pred[0])
    # if max > 5:
    label = num_to_label[np.argmax(pred[0])]
    # else:
    #     label = "NO POSE"
    return label

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
  img = utils.draw_prediction_on_image(image, person, crop_region=None, 
                              close_figure=False, keep_input_size=True)
  return [img, embeddings]

def img_with_label(img, label: str):
   img_with_rect = cv2.rectangle(img, (0, 0), (300, 80), (0, 0, 0), -1)
   return cv2.putText(img_with_rect, label, (10, 50), fontScale=2, fontFace=cv2.FONT_HERSHEY_SIMPLEX, thickness=3, color=(255, 255, 255))

def live_camera_loop(): 
  # define a video capture object
  vid = cv2.VideoCapture(0)
    
  while(True):
      # Capture the video frame
      ret, frame = vid.read()
      prediction = predict_label(frame, person_detect_model, person_detect_datasets)
      print(prediction)
      if (prediction == "1"):
        # go to movenet / pose detection pipeline
        print("PERSON")
        img, embeddings = image_prediction(frame)
        print(img.shape)
        label = predict_pose(embeddings)
        img = img_with_label(img, label)

         # feed embeddings into our pose detection model with predict label
         
         # output result
      else:
         print("no person")
         img = frame
    
      # Display the resulting frame
      cv2.imshow('frame', img)
        
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
#   test_image_url = "./data/test/downdog/00000002.jpg"
#   prediction = predict_label(test_image_url, person_detect_model, person_detect_datasets)
#   print(prediction)

  live_camera_loop()
    
main()