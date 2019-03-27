import os
import pickle
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
from torchvision import transforms


class CNN(nn.Module):
    def __init__(self, name):
        super(CNN, self).__init__()
        self.name = name
        # TODO modify the layers
        self.conv1 = nn.Conv2d(3, 16, 3, stride=2, padding=0)
        self.conv2 = nn.Conv2d(16, 32, 3, stride=2, padding=0)

        # self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        # self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(11 * 11 * 32, 256)
        self.fc2 = nn.Linear(256, 6)
        # self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        # TODO modify the layers
        x = F.tanh(self.conv1(x))
        #print("after first",x.shape)
        x = F.tanh(self.conv2(x))
        #print("after second",x.shape)
        # x = self.pool(x)
        # x = F.relu(self.conv3(x))
        # x = F.relu(self.conv4(x))
        #x = self.pool(x)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        #print("after 3rd",x.shape)
        x = self.fc2(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
