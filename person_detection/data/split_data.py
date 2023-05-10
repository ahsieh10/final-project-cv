import os
import random
import shutil

# Assuming final-project-csv folder is open
all0_dir = 'all0'
all1_dir = 'all1'

train_dir = 'train'
test_dir = 'test'

class_0 = '0'
class_1 = '1'

# Define the train/test split ratio
split_ratio = 0.6

# Create the train and test directories if they don't already exist
os.makedirs(os.path.join(train_dir, class_0), exist_ok=True)
os.makedirs(os.path.join(train_dir, class_1), exist_ok=True)
os.makedirs(os.path.join(test_dir, class_0), exist_ok=True)
os.makedirs(os.path.join(test_dir, class_1), exist_ok=True)

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

split_dir(all0_dir, class_0)
split_dir(all1_dir, class_1)