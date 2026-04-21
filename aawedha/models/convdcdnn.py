# Implements ConvDCDNN model from:
# Zhang, Z., He, Y., Mai, W., Luo, Y., Li, X., Cheng, Y., Huang, X. and Lin, R. (2025) 
# ‘Convolutional Dynamically Convergent Differential Neural Network for Brain Signal Classification’, 
# IEEE Transactions on Neural Networks and Learning Systems, 36(5), pp. 8166–8177. 
# Available at: https://doi.org/10.1109/TNNLS.2024.3437676.
# Adapted from official implementation: https://github.com/BIRLab/ConvDCDNN


import torch
from torch import nn
import math


def conv_output_size(input_size, kernel_size, stride, padding, dilation=1):
    """According to https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html"""
    return math.floor((input_size + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)


class StimulusClassifier(nn.Module):
    def __init__(self, nb_classes=2, Chans=15, Samples=512, 
                 fusion_channels= 32, 
                 kernel_size=15, 
                 kernel_stride=15, 
                 dropout=0.25,
                 return_features=False,
                 name="ConvDCDNN"):
        """
        Initialize a stimulus classifier

        :param num_channels: EEG channels number
        :param num_samples: EEG samples number (i.e., time window size)
        :param fusion_channels: output channels number of convolutional layer
        :param kernel_size: convolutional kernel size
        :param kernel_stride: convolutional kernel stride
        :param dropout: dropout rate
        """
        super().__init__()
        self.name = name
        self.return_features = return_features
        self.conv = nn.Sequential(
            nn.Conv1d(Chans, fusion_channels, kernel_size, kernel_stride, 0),
            nn.Dropout(dropout),
            nn.Tanh()
        )
        self.output = nn.Sequential(
            nn.Flatten(),
            nn.Linear(fusion_channels * conv_output_size(Samples, kernel_size, kernel_stride, 0), nb_classes)
        )

    def forward(self, x):
        """
        Predict P300 signal

        :param x: input tensor (batch, channel, sample)
        :return: predictions (batch, 2)
        """
        x = self.conv(x)
        x = self.output(x)
        return x

