# -*- coding: utf-8 -*-

import torch
import numpy as np

class LogTransform(object):
    """
    Perform log-amplitude transform on the data
    
    Args:
        factor (int): scale of the Gaussian noise. default: 1e-5
    """

    def __init__(self, clip=1e-3):
        self.clip = clip

    def __call__(self, data):
        if (self.clip == 0):
            data = torch.log1p(data)
        else:
            data = torch.log(data + self.clip)
        return data

class NormalizeTensor(object):
    """
    Normalize an tensor with given mean (M1,...,Mn) and std (S1,..,Sn) 
    for n channels. This transform will normalize each channel of the input 
    input[channel] = (input[channel] - mean[channel]) / std[channel]
    
    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        """
        Args:
            data (Numpy array): Data of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized data
        """
        if (type(self.mean) == float):
            data = (data - self.mean) / self.std
        else:
            for c in range(data.shape[0]):
                data[c] = (data[c] - self.mean[c]) / self.std[c]
        return data   

class NoiseGaussian(object):
    """
    Adds gaussian noise to a given matrix.
    
    Args:
        factor (int): scale of the Gaussian noise. default: 1e-5
    """

    def __init__(self, factor=1e-5):
        self.factor = factor

    def __call__(self, data):
        """
        Args:
            tensor (Tensor): Tensor of data
        Returns:
            Tensor: Noisy tensor with additive Gaussian noise
        """
        data = data + (torch.randn_like(data) * self.factor)
        return data
    
class OutliersZeroRandom(object):
    """
    Randomly add zeroed-out outliers (without structure)
    
    Args:
        factor (int): Percentage of outliers to add. default: .25
    """

    def __init__(self, factor=.2):
        self.factor = factor

    def __call__(self, data):
        """
        Args:
            tensor (Tensor): Tensor of data
        Returns:
            Tensor: Tensor with randomly zeroed-out outliers
        """
        # Add random outliers (here similar to dropout mask)
        ones = torch.rand_like(data) > self.factor
        data = data * ones.float()
        return data

class MaskRows(object):
    """
    Put random rows to zeros
    
    Args:
        factor (int): Percentage to be put to zero. default: .2
    """

    def __init__(self, factor=.2):
        self.factor = factor

    def __call__(self, data):
        """
        Args:
            data (Numpy array): Matrix to be masked
        Returns:
            Numpy array: Masked matrix.
        """
        data = data.copy()
        tmpIDs = np.floor(np.random.rand(int(np.floor(data.shape[0] * self.factor))) * (data.shape[0]))
        for i in range(tmpIDs.shape[0]):
            if tmpIDs[i] < data.shape[0]:
                data[int(tmpIDs[i]), :] = 0
        return data

class MaskColumns(object):
    """
    Put random columns to zeros
    
    Args:
        factor (int): Percentage to be put to zero. default: .2
    """

    def __init__(self, factor=.2):
        self.factor = factor

    def __call__(self, data):
        """
        Args:
            data (Numpy array): Matrix to be masked
        Returns:
            Numpy array: Masked matrix.
        """
        data = data.copy()
        tmpIDs = np.floor(np.random.rand(int(np.floor(data.shape[1] * self.factor))) * (data.shape[1]))
        for i in range(tmpIDs.shape[0]):
            if tmpIDs[i] < data.shape[1]:
                data[:, int(tmpIDs[i])] = 0
        return data