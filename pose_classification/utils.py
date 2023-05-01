import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import torch
from torch.utils.data import DataLoader, Dataset


class WineDataset(Dataset):
    def __init__(self, X, Y):
        """
        Initialize the wine dataset variables.
        :param X: Features of shape [n, 11], where n is number of samples.
        :param Y: Labels of shape [n], where n is number of samples.
        """
        self.X = X
        self.Y = Y

    def __len__(self):
        """
        Returns the number of samples.
        :return: The number of samples.
        """
        return self.X.shape[0]

    def __getitem__(self, index):
        """
        Returns feature and label of the sample at the given index.
        :param index: Index of a sample.
        :return: Feature and label of the sample at the given index.
        """
        return self.X[index], torch.Tensor([self.Y[index]])


class MNISTDataset(Dataset):
    def __init__(self, X, Y, height=8, width=8):
        """
        Initialize the MNIST dataset variables.
        :param X: Features of shape [n, 64], where n is number of samples.
        :param Y: Labels of shape [n], where n is number of samples.
        :param height: Height of each image.
        :param width: Width of each image.
        """
        self.X = X
        self.Y = torch.LongTensor(Y)

        # Shape of an image
        self.h = height
        self.w = width

    def __len__(self):
        """
        Returns the number of samples.
        :return: The number of samples.
        """
        return self.X.shape[0]

    def __getitem__(self, index):
        """
        Reshapes a sample feature of shape [64] to [1, self.h, self.w].
        Returns the reshaped feature and label of the sample at the given index.
        :param index: Index of a sample.
        :return: Reshaped feature and label of the sample at the given index.
        """
        x = self.X[index]
        # TODO: Reshape vector x to [1, self.h, self.w]
        x= x.reshape((1, self.h, self.w))

        return x, self.Y[index]


def get_wine_loader(batch_size, dataset="data/wine.txt", test_size=0.2):
    """
    Returns train/test dataloaders of UCI Wine dataset.
    :param batch_size: Batch size.
    :param dataset: Path to UCI Wine dataset wine.txt.
    :param test_size: Ratio of (test set size / dataset size)
    :return: Dataloaders for training set and test set.
    """
    # Check if the file exists
    if not os.path.exists(dataset):
        raise FileNotFoundError('The file {} does not exist'.format(dataset))

    # Load the dataset
    data = np.loadtxt(dataset, skiprows=1)
    X, Y = data[:, 1:], data[:, 0]

    # Normalize the features
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

    # Convert to float32
    X = X.astype(np.float32)
    Y = Y.astype(np.float32)

    # Split data into training set and test set
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size)

    # Build dataset
    dataset_train = WineDataset(X_train, Y_train)
    dataset_test = WineDataset(X_test, Y_test)

    # Build dataloader
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

    return dataloader_train, dataloader_test


def get_mnist_loader(batch_size, dataset="data/digits.csv", test_size=0.2,
                     shuffle_train_label=False, shuffle_ratio=0.8):
    """
    Returns train/test dataloaders of MNIST dataset.
    :param batch_size: Batch size.
    :param dataset: Path to MNIST dataset digits.csv.
    :param test_size: Ratio of (test set size / dataset size)
    :param shuffle_train_label: If True, shuffle the training labels.
        Useful for the third report question.
    :param shuffle_ratio: Proportion of training labels to shuffle.
    :return: Dataloaders for training set and test set.
    """
    # Check if the file exists
    if not os.path.exists(dataset):
        raise FileNotFoundError('The file {} does not exist'.format(dataset))

    # Read data
    data = pd.read_csv("data/digits.csv", header=0)

    # We assume labels are in the first column of the dataset
    Y = data.values[:, 0].astype(np.int32)

    # Features columns are indexed from 1 to the end, make sure that dtype = float32
    X = data.values[:, 1:].astype(np.float32)

    # Split data into training set and test set
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size)

    # Shuffle training label
    if shuffle_train_label:
        shuffle(Y_train, shuffle_ratio)

    # Build dataset
    dataset_train = MNISTDataset(X_train, Y_train)
    dataset_test = MNISTDataset(X_test, Y_test)

    # Build dataloader
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

    return dataloader_train, dataloader_test


def shuffle(Y, shuffle_ratio):
    """
    Shuffles a numpy array in-place.
    :param Y: 1D numpy array to shuffle.
    :param shuffle_ratio: Proportion of training labels to shuffle.
    :return: None
    """
    shuffles_size = int(shuffle_ratio * Y.shape[0])
    for _ in range(10):
        np.random.shuffle(Y[:shuffles_size])


def visualize_loss(losses):
    """
    Uses Matplotlib to visualize loss per batch.
    :return: None
    """
    x = np.arange(1, len(losses) + 1)
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.title('Training Loss per Epoch')
    plt.plot(x, losses)
    plt.show()


def visualize_accuracy(accuracies):
    """
    Uses Matplotlib to visualize accuracy per batch.
    :return: None
    """
    x = np.arange(1, len(accuracies) + 1)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy per Epoch')
    plt.plot(x, accuracies)
    plt.show()


def visualize_image(model, dataloader, row=4, col=4):
    """
    Credit: CS1470/2470
    Uses Matplotlib to visualize the results of our CNN model.
    :param model: CNN model.
    :param dataloader: MNIST test set Dataloader.
    :param row: Number of rows to plot.
    :param col: Number of columns to plot.
    :return: None
    """
    image_num = row * col
    images = []
    labels = []
    predictions = []
    # Get images, labels, and predictions
    model.eval()
    with torch.no_grad():
        for X, Y in dataloader:
            out = model(X)
            predictions.extend(torch.argmax(out, 1)[:image_num - len(images)].detach().cpu().numpy())
            labels.extend(Y[:image_num - len(images)].detach().cpu().numpy())
            images.extend(map(lambda t: t[0], X[:image_num - len(images)].detach().cpu().numpy()))
            if len(images) == image_num:
                break
    plot_grid_common(images, predictions, labels, row, col)


def plot_grid_common(images, predictions, labels, row, col):
    """
    Plots image grid and labels.
    :param images: A list of images.
    :param predictions: A list of prediction categories.
    :param labels: A list of ground truth categories.
    :param row: Number of rows to plot.
    :param col: Number of columns to plot.
    :return: None
    """
    # Plot
    fig, axes = plt.subplots(row, col)
    fig.suptitle("g.t. = ground truth")
    image_id = 0
    for i in range(row):
        for j in range(col):
            ax = axes[i][j]
            if image_id < len(images):
                ax.imshow(images[image_id], cmap="Greys")
                ax.set(title="predict: {}\ng.t.: {}".format(predictions[image_id], labels[image_id]))
                plt.setp(ax.get_xticklabels(), visible=False)
                plt.setp(ax.get_yticklabels(), visible=False)
                ax.tick_params(axis='both', which='both', length=0)
            else:
                ax.remove()
            image_id += 1
    plt.show()


def visualize_misclassified_image(model, dataloader, row=4, col=4):
    """
    Uses Matplotlib to visualize the misclassified images of our CNN model.
    :param model: CNN model.
    :param dataloader: MNIST test set Dataloader.
    :param row: Number of rows to plot.
    :param col: Number of columns to plot.
    :return: None
    """
    image_num = row * col
    images = []
    labels = []
    predictions = []
    # Get images, labels, and predictions of misclassified images
    model.eval()
    with torch.no_grad():
        for X, Y in dataloader:
            out = model(X)
            pred_classes = torch.argmax(out, 1)
            diff_ids = pred_classes != Y
            predictions.extend(pred_classes[diff_ids][:image_num - len(images)])
            labels.extend(Y[diff_ids][:image_num - len(images)])
            images.extend(map(lambda t: t[0], X[diff_ids][:image_num - len(images)].detach().cpu().numpy()))
            if len(images) == image_num:
                break
    plot_grid_common(images, predictions, labels, row, col)


def visualize_confusion_matrix(model, dataloader, class_num=10):
    """
    Plot confusion matrix.
    The intersection of the i-th row and j-th column cm[i][j] means the number of images
    that are predicted as class j while the ground truth label is i.
    Reference: https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html
    :param model: CNN model.
    :param dataloader: MNIST test set Dataloader.
    :param class_num: Number of categories, 10 for MNIST dataset.
    :return: None
    """
    predictions = []
    labels = []
    digits = np.arange(class_num)
    # Get predictions and labels
    model.eval()
    with torch.no_grad():
        for X, Y in dataloader:
            out = model(X)
            prediction = torch.argmax(out, 1)
            labels.extend(Y.detach().cpu().numpy())
            predictions.extend(prediction.detach().cpu().numpy())
    cm = confusion_matrix(y_true=labels, y_pred=predictions, labels=digits)

    fig, ax = plt.subplots()
    ax.imshow(cm)

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(digits, digits)
    ax.set_yticks(digits, digits)

    # Loop over data dimensions and create text annotations
    for i in digits:
        for j in digits:
            ax.text(j, i, cm[i, j], ha="center", va="center", color="w")

    ax.set_title("Confusion Matrix on MNIST Test Set")
    ax.set_xlabel("Prediction")
    ax.set_ylabel("Ground Truth")
    fig.tight_layout()
    plt.show()
