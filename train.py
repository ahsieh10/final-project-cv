import cv2
import os
import numpy as np
from movenet.movenet import Movenet
import movenet.utils as utils
from pose_classification_2.run import train

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

def load_data():
    absolute_path = os.path.dirname(__file__)
    relative_path = './data/train'
    full_path = os.path.join(absolute_path, relative_path)
    train_data = []
    train_labels = []
    test_data = []
    test_labels = []
    for label in os.listdir(full_path):
        if label != '.DS_Store':
            sub_path = os.path.join(full_path, label)
            for img_path in os.listdir(sub_path):
                if img_path != '.DS_Store':
                    print(sub_path, img_path, label)
                    img = cv2.imread(os.path.join(sub_path, img_path))
                    person = detect(img)
                    embedding = utils.get_embedding(person)
                    train_data += [embedding]
                    train_labels += label
    train_array = np.array(train_data)
    train_label_array = np.array(train_labels)
    full_path = os.path.join(absolute_path, './data/test')
    for label in os.listdir(full_path):
        if label != '.DS_Store':
            sub_path = os.path.join(full_path, label)
            for img_path in os.listdir(os.path.join(full_path, label)):
                if img_path != '.DS_Store':
                    print(sub_path, img_path)
                    img = cv2.imread(os.path.join(sub_path, img_path))
                    person = detect(img)
                    embedding = utils.get_embedding(person)
                    test_data += [embedding]
                    test_labels += label
    test_array = np.array(test_data)
    test_label_array = np.array(test_labels)
    return train_array, train_label_array, test_array, test_label_array

train_data, train_label, test_data, test_label = load_data()

train(train_data, train_label, test_data, test_label)