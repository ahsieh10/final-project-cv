import os
import random
import numpy as np
from PIL import Image
import tensorflow as tf

from .hyperparameters import img_size, num_classes, preprocess_sample_size, batch_size

class Datasets():
    """ Class for containing the training and test sets as well as
    other useful data-related information. Contains the functions
    for preprocessing.
    """

    def __init__(self, data_path, task):

        self.data_path = data_path
        self.task = task

        # Dictionaries for (label index) <--> (class name)
        self.idx_to_class = {}
        self.class_to_idx = {}

        # For storing list of classes
        self.classes = [""] * num_classes

        # Mean and std for standardization
        self.mean = np.zeros((img_size,img_size,3))
        self.std = np.ones((img_size,img_size,3))
        self.calc_mean_and_std()

        # Setup data generators
        # These feed data to the training and testing routine based on the dataset
        self.train_data = self.get_data(
            os.path.join(self.data_path, "train/"), task == '3', True, True)
        self.test_data = self.get_data(
            os.path.join(self.data_path, "test/"), task == '3', False, False)

    def calc_mean_and_std(self):
        """ Calculate mean and standard deviation of a sample of the
        training dataset for standardization.
        Arguments: none
        Returns: none
        """

        # Get list of all images in training directory
        file_list = []
        for root, _, files in os.walk(os.path.join(self.data_path, "train/")):
            for name in files:
                if name.endswith(".jpg"):
                    file_list.append(os.path.join(root, name))

        # Shuffle filepaths
        random.shuffle(file_list)

        # Take sample of file paths
        file_list = file_list[:preprocess_sample_size]

        # Allocate space in memory for images
        data_sample = np.zeros(
            (preprocess_sample_size, img_size, img_size, 3))

        # Import images
        for i, file_path in enumerate(file_list):
            img = Image.open(file_path)
            img = img.resize((img_size, img_size))
            img = np.array(img, dtype=np.float32)
            img /= 255.

            # Grayscale -> RGB
            if len(img.shape) == 2:
                img = np.stack([img, img, img], axis=-1)

            data_sample[i] = img[:, :, :3]

        self.mean = np.mean(data_sample, 0)
        self.std = np.std(data_sample, 0)

        print("Dataset mean shape: [{0}, {1}, {2}]".format(
            self.mean.shape[0], self.mean.shape[1], self.mean.shape[2]))

        print("Dataset mean top left pixel value: [{0:.4f}, {1:.4f}, {2:.4f}]".format(
            self.mean[0,0,0], self.mean[0,0,1], self.mean[0,0,2]))

        print("Dataset std shape: [{0}, {1}, {2}]".format(
            self.std.shape[0], self.std.shape[1], self.std.shape[2]))

        print("Dataset std top left pixel value: [{0:.4f}, {1:.4f}, {2:.4f}]".format(
            self.std[0,0,0], self.std[0,0,1], self.std[0,0,2]))

    def standardize(self, img):
        """ Function for applying standardization to an input image.
        Arguments:
            img - numpy array of shape (image size, image size, 3)
        Returns:
            img - numpy array of shape (image size, image size, 3)
        """

        img = (img - self.mean) / self.std 

        return img

    def preprocess_fn(self, img):
        """ Preprocess function for ImageDataGenerator. """

        if self.task == '3':
            img = tf.keras.applications.vgg16.preprocess_input(img)
        else:
            img = img / 255.
            img = self.standardize(img)
        return img

    def get_data(self, path, is_vgg, shuffle, augment):
        """ Returns an image data generator which can be iterated
        through for images and corresponding class labels.
        Arguments:
            path - Filepath of the data being imported, such as
                   "../data/train" or "../data/test"
            is_vgg - Boolean value indicating whether VGG preprocessing
                     should be applied to the images.
            shuffle - Boolean value indicating whether the data should
                      be randomly shuffled.
            augment - Boolean value indicating whether the data should
                      be augmented or not.
        Returns:
            An iterable image-batch generator
        """

        if augment:
            data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
                    preprocessing_function=self.preprocess_fn,
                    rotation_range=10,
                    width_shift_range=0.1,
                    height_shift_range=0.1,
                    zoom_range=0.1,
                )
        else:
            data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
                preprocessing_function=self.preprocess_fn)

        im_size = 224 if is_vgg else img_size
        classes_for_flow = None

        # Make sure all data generators are aligned in label indices
        if bool(self.idx_to_class):
            classes_for_flow = self.classes
        print(path)
        # Form image data generator from directory structure
        data_gen = data_gen.flow_from_directory(
            path,
            target_size=(im_size, im_size),
            class_mode='sparse',
            batch_size=batch_size,
            shuffle=shuffle,
            classes=classes_for_flow)

        # Setup the dictionaries if not already done
        if not bool(self.idx_to_class):
            unordered_classes = []
            for dir_name in os.listdir(path):
                if os.path.isdir(os.path.join(path, dir_name)):
                    unordered_classes.append(dir_name)

            for img_class in unordered_classes:
                self.idx_to_class[data_gen.class_indices[img_class]] = img_class
                self.class_to_idx[img_class] = int(data_gen.class_indices[img_class])
                self.classes[int(data_gen.class_indices[img_class])] = img_class

        return data_gen