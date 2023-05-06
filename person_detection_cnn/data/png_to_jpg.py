import os
import random
import shutil
import cv2

def convert_to_jpg(src_dir: str):
    # Get a list of the filenames in your all0 directory
    all_filenames = os.listdir(src_dir)
    for filename in all_filenames:
        src = os.path.join(src_dir, filename)
        image = cv2.imread(src)
        # print(filename)
        # print(src_dir + "-jpg/" + filename[:-4] + ".jpg")
        cv2.imwrite(src_dir + "-jpg/" + filename[:-4] + ".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

convert_to_jpg("./1")