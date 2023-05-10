'''Entrypoint for our application'''

import cv2
import numpy as np
import tensorflow as tf
import os

import movenet.utils as utils
from movenet.movenet import Movenet
from person_detection.code.models import YourModel as PDModel
from person_detection.code.models import VGGModel as PDVGG
from person_detection.code.preprocess import Datasets as PDDatasets
from person_detection.code.predict import predict_label
from util.constants import num_to_label


movenet = Movenet('movenet/lite-model_movenet_singlepose_thunder_tflite_float16_4')

'''Load human detection model'''
person_detect_vgg_weights_path = "./person_detection/code/checkpoints/vgg/vgg.weights.e012-acc0.9692.h5"
person_detect_vgg_base_path = "./person_detection/code/vgg16_imagenet.h5"
person_detect_weights_path = "./person_detection/code/checkpoints/your_model/your.weights.e043-acc0.9205.h5"

person_detect_vgg_model = PDVGG()
person_detect_model = PDModel()

person_detect_vgg_model(tf.keras.Input(shape=(224, 224, 3)))
person_detect_model(tf.keras.Input(shape=(224, 224, 3)))

person_detect_vgg_model.vgg16.load_weights(person_detect_vgg_base_path, by_name=True)
person_detect_vgg_model.head.load_weights(person_detect_vgg_weights_path, by_name=False)
person_detect_model.load_weights(person_detect_weights_path)

person_detect_datasets = PDDatasets('./person_detection'+os.sep+'data'+os.sep, "TASK 1")

'''Load pose classifier model'''

pose_weights_path = "./pose_classification/weights/best_weights_updated.h5"

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
    label = num_to_label[np.argmax(pred[0])]
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
  is_vgg_model = False
    
  while(True):
      # Capture the video frame
      ret, frame = vid.read()
      if (is_vgg_model):
        prediction = predict_label(frame, person_detect_vgg_model, person_detect_datasets)
      else:
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
      pressed = cv2.waitKey(1) & 0xFF 
      if pressed == ord("m"):
        is_vgg_model = not is_vgg_model
        print(is_vgg_model)
      elif pressed == ord("q"):
          break
    
  # After the loop release the cap object
  vid.release()
  # Destroy all the windows
  cv2.destroyAllWindows()

def main():
  live_camera_loop()
    
main()