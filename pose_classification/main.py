from models import OneLayerNN, TwoLayerNN, CNN, train, test, correct_predict_num
from utils import get_mnist_loader, get_wine_loader, \
    visualize_loss, visualize_accuracy, visualize_image, visualize_confusion_matrix, visualize_misclassified_image

import numpy as np
import random

import torch
from torch import nn, optim


def test_linear_nn(test_size=0.2):
    """
    Tests TwoLayerNN on the yoga dataset.
    :param test_size: The ratio of test set w.r.t. the whole dataset.
    :return: Loss on test set.
    """

    # TODO: Tune these hyper-parameters
    # Hyper-parameters of TwoLayerNN
    batch_size = 15  # batch size
    num_epoch = 50  # number of training epochs
    learning_rate = 0.05  # learning rate

    # Load data
    dataloader_train, dataloader_test = get_wine_loader(batch_size=batch_size, test_size=test_size)

    # Initialize model
    model = TwoLayerNN(input_features=11)
    # TODO: Initialize optimizer
    optimizer = optim.SGD(model.parameters() ,lr=learning_rate)

    # TODO: Initialize the MSE (i.e., L2) loss function
    loss_func = nn.MSELoss()

    losses = train(model, dataloader_train, loss_func, optimizer, num_epoch)

    # Uncomment to visualize training losses
    # visualize_loss(losses)

    # Average training/testing loss
    loss_train = test(model, dataloader_train, loss_func)
    loss_test = test(model, dataloader_test, loss_func)
    print('Average Training Loss:', loss_train)
    print('Average Testing Loss:', loss_test)

    return loss_test


def main():
    
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    # Uncomment to test your models
    test_linear_nn()


if __name__ == "__main__":
    main()
