import os
import random
import shutil

train_dir = 'train'
test_dir = 'test'

new_classes = ["chair", "cobra", "mountain", "warrior"]

# Define the train/test split ratio
split_ratio = 0.66

for label in new_classes:
    os.makedirs(os.path.join(train_dir, label), exist_ok=True)
    os.makedirs(os.path.join(test_dir, label), exist_ok=True)

def split_dir(src_dir: str, class_num: str):
    # Get a list of the filenames in your all0 directory
    all_filenames = os.listdir(src_dir)

    # Randomly shuffle the filenames
    random.shuffle(all_filenames)

    # Calculate the split point between train and test
    split_point = int(split_ratio * len(all_filenames))

    for filename in all_filenames[:split_point]:
        src = os.path.join(src_dir, filename)
        dst = os.path.join(train_dir, class_num, filename)
        shutil.copyfile(src, dst)

    for filename in all_filenames[split_point:]:
        src = os.path.join(src_dir, filename)
        dst = os.path.join(test_dir, class_num, filename)
        shutil.copyfile(src, dst)

for label in new_classes:
    split_dir(label, label)