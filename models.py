from __future__ import annotations

from typing import Callable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

# Used Deep Learning HW (11) Code for several components of the models/training/testing
class Linear(nn.Module):
    def __init__(self, input_features: int):
        """
        Initializes one linear layer.

        Parameters
        ----------
        input_features : int, default=11
            The number of features of each sample.
        """
        super().__init__()

        self.weights = torch.nn.Linear(input_features, 3) # 3 classes, so output size of 3

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Applies the linear layer defined in the constructor to input features X.

        Parameters
        ----------
        X : torch.Tensor
            2D torch tensor of shape [n, 11], where n is batch size.
            Represents features of a batch of data.

        Returns
        -------
        torch.Tensor
            2D torch tensor of shape [n, 1], where n is batch size.
            Represents prediction of wine quality.
        """

        return self.weights(X)
        pass



class Nonlinear_Net(nn.Module):
   def __init__(self, input_size, hidden_size, num_classes):
       super(Nonlinear_Net, self).__init__()
       self.layer1 = nn.Linear(input_size, hidden_size)
       self.relu1 = nn.ReLU()
       self.layer2 = nn.Linear(hidden_size, num_classes)

   def forward(self, x):
       res = self.layer1(x)
       res = self.relu1(res)
       res = self.layer2(res)
       return res


def train(
    model: nn.Module,
    dataloader: DataLoader,
    loss_func: nn.CrossEntropyLoss,
    optimizer: torch.optim,
    num_epoch: int,
    correct_num_func: Callable = None,
    print_info: bool = True,
) -> list[float] | tuple[list[float], list[float]]:
    # Initialize an empty list to save average losses in all epochs.

    epoch_average_losses = []
    # Initialize an empty list to save accuracy values in all epochs.
    average_accuracy_values = []
    # Tell the model we are in the training phase.
    model.train()
    # This is useful if you use batch normalization or dropout
    # because the behavior of these layers in the training phase is different from testing phase.

    # train network for num_epochs
    for epoch in range(num_epoch):
        # Initializing variables
        epoch_loss_sum = 0
        # Sum of the number of correct predictions. Will be used to calculate average accuracy for CNN.
        epoch_correct_num = 0

        # Iterate through batches.

        for X, Y in dataloader:

            outputs = model(X)

            # n = torch.argmax(outputs, dim=1)
            # n = n.view(n.size(0), 1)

            # debug = Y.squeeze()
            loss = loss_func(outputs, Y.long().squeeze())

            optimizer.zero_grad()

            # loss.requires_grad = True
            loss.backward()

            optimizer.step()

            epoch_loss_sum += loss.item() * X.shape[0]

            if correct_num_func != None:
                epoch_correct_num += correct_predict_num(outputs, Y)

        epoch_average_losses.append(epoch_loss_sum / len(dataloader.dataset))

        if correct_num_func:
            avg_acc = epoch_correct_num / len(dataloader.dataset)
            average_accuracy_values.append(avg_acc)

        # Print the loss after every epoch. Print accuracies if specified
        # print_info = True
        print_info = False
        if print_info:
            print(
                "Epoch: {} | Loss: {:.4f} ".format(
                    epoch, epoch_loss_sum / len(dataloader.dataset)
                ),
                end="",
            )
            if correct_num_func:
                print(
                    "Accuracy: {:.4f}%".format(
                        epoch_correct_num / len(dataloader.dataset) * 100
                    ),
                    end="",
                )
            print()

    if correct_num_func is None:
        return epoch_average_losses
    else:
        return epoch_average_losses, average_accuracy_values

def test(
    model: nn.Module,
    dataloader: DataLoader,
    loss_func: nn.CrossEntropyLoss,
    correct_num_func: Callable = None,
) -> float | tuple[float, float]:
    """
    Tests the model.

    Parameters
    ----------
    model : torch.nn.Module
        A deep model.
    dataloader : torch.utils.data.DataLoader
        Dataloader of the testing set. Contains the testing data equivalent to ((Xi, Yi)),
        where (Xi, Yi) is a batch of data.
        X: 2D torch tensor for UCI wine and 4D torch tensor for MNIST.
        Y: 2D torch tensor for UCI wine and 1D torch tensor for MNIST, containing the corresponding labels
            for each example.
        Refer to the Data Format section in the handout for more information.
    loss_func : torch.nn.MSELoss
        An MSE loss function for UCI wine and a cross entropy loss for MNIST.
    correct_num_func : Callable, default=None
        A function to calculate how many samples are correctly classified.
        You need to implement correct_predict_num() below.
        To test the CNN model, we also want to calculate the classification accuracy in addition to loss.

    Returns
    -------
    float
        Average loss.
    float
        Average accuracy. This is applicable when testing on MNIST.
    """

    epoch_loss_sum = 0
    epoch_correct_num = 0

    model.eval()
    with torch.no_grad():
        for X, Y in dataloader:
            outputs = model(X)
            loss = loss_func(outputs, Y.long().squeeze())

            epoch_loss_sum += loss.item() * X.shape[0]
            if correct_num_func:
                epoch_correct_num += correct_predict_num(outputs, Y)
    avg_epoch_loss = epoch_loss_sum / len(dataloader.dataset)
    avg_epoch_accuracy = epoch_correct_num / len(dataloader.dataset)

    if correct_num_func is None:
        return avg_epoch_loss
    else:
        return avg_epoch_loss, avg_epoch_accuracy

    pass


def correct_predict_num(logit: torch.Tensor, target: torch.Tensor) -> float:
    """
    Returns the number of correct predictions.

    Parameters
    ----------
    logit : torch.Tensor
        2D torch tensor of shape [n, class_num], where
        n is the number of samples, and class_num is the number of classes (10 for MNIST).
        Represents the output of CNN model.
    target : torch.Tensor
        1D torch tensor of shape [n],  where n is the number of samples.
        Represents the ground truth categories of images.

    Returns
    -------
    float
        A python scalar. The number of correct predictions.
    """
    n = torch.argmax(logit, dim=1)
    total = 0
    for i in range(len(n)):
        if (n[i] == target[i].item()):
            total += 1
    return total