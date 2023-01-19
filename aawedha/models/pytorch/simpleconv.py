# Simple ConvNet 
from aawedha.models.pytorch.torch_inits import initialize_Glorot_uniform
from aawedha.models.pytorch.torchdata import reshape_input
import torch.nn.functional as F
import torch.nn as nn
from torch import flatten


class SimpleTorch(nn.Module):

    def __init__(self):
        super().__init__()
        self.name = "SimpleConv"
        # self.conv = nn.Conv2d(1, 16, kernel_size=8, padding='same')
        self.conv = nn.Conv2d(1, 16, kernel_size=8)
        self.bnorm = nn.BatchNorm2d(16)
        self.drop = nn.Dropout(p=0.5)
        self.fc = nn.Linear(16*8*198, 1)
        initialize_Glorot_uniform(self)

    def forward(self, x):
        x = reshape_input(x)
        # x = F.pad(x, (0, 0, 6, 6))  # [left, right, top, bot]
        x = F.elu(self.bnorm(self.conv(x)))
        # print(x.shape)
        x = self.drop(x)
        x = flatten(x, 1)
        # print(x.shape)
        x = self.fc(x)
        return x
