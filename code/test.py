import os
import cv2

#for filename in listdir('C:/tensorflow/models/research/object_detection/images/train'):
for root, _, files in os.walk(os.path.join("../data", "train/")):
    for name in files:
        if name.endswith(".jpg"):
            print(os.path.join(root, name))
            cv2.imread(os.path.join(root, name))