import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Tuple


class PremDataset(Dataset):
    def __init__(self, X: np.ndarray, Y: np.ndarray):
        """
        Initialize the wine dataset variables.

        Parameters
        ----------
        X : np.ndarray
            Features of shape [n, 11], where n is number of samples.
        Y : np.ndarray
            Labels of shape [n], where n is number of samples.
        """
        self.X = X
        self.Y = Y

    def __len__(self) -> int:
        """
        Returns the number of samples.

        Returns
        -------
        int
            The number of samples.
        """
        return self.X.shape[0]

    def __getitem__(self, index: int) -> Tuple[np.ndarray, torch.Tensor]:
        """
        Returns feature and label of the sample at the given index.

        Parameters
        ----------
        index : int
            Index of a sample.

        Returns
        -------
        tuple[np.ndarray, torch.Tensor]
            Feature and label of the sample at the given index.
        """
        return self.X[index], torch.Tensor([self.Y[index]])

