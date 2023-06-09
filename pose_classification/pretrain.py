import cv2
import os
import numpy as np
from movenet.movenet import Movenet
import movenet.utils as utils
from util.constants import label_to_num
import pandas as pd

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
    relative_path = './pose_data/train'
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
                    img = cv2.imread(os.path.join(sub_path, img_path))
                    if (img is not None):
                        label = label.strip()
                        person = detect(img)
                        embedding = utils.get_embedding(person)
                        train_data += [embedding]
                        train_labels.append(label_to_num[label])
    train_array = np.array(train_data)
    train_label_array = np.array(train_labels)
    full_path = os.path.join(absolute_path, './pose_data/test')
    for label in os.listdir(full_path):
        if label != '.DS_Store':
            sub_path = os.path.join(full_path, label)
            for img_path in os.listdir(os.path.join(full_path, label)):
                if img_path != '.DS_Store':
                    img = cv2.imread(os.path.join(sub_path, img_path))
                    if (img is not None):
                        label = label.strip()
                        person = detect(img)
                        embedding = utils.get_embedding(person)
                        test_data += [embedding]
                        test_labels.append(label_to_num[label])
    test_array = np.array(test_data)
    test_label_array = np.array(test_labels)
    return train_array, train_label_array, test_array, test_label_array

def data_to_csv(train_data, train_label, test_data, test_label):
    train_data_with_labels = np.concatenate((train_data, np.expand_dims(train_label, 1)), axis=1)
    test_data_with_labels = np.concatenate((test_data, np.expand_dims(test_label, 1)), axis=1)
    df = pd.DataFrame(train_data_with_labels)
    df.to_csv("./csv_data/train_data_updated.csv", index=False)
    df2 = pd.DataFrame(test_data_with_labels)
    df2.to_csv("./csv_data/test_data_updated.csv", index=False)
    
def convert_data():
    train_data, train_label, test_data, test_label = load_data()
    data_to_csv(train_data, train_label, test_data, test_label)

if __name__ == "__main__":
    convert_data()