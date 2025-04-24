import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.optim as optim

class settings:
    things = 2

class neuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Input layer 3 because of 3 channels.
        # Out channels is our depth we want to convert to. 256 filters each 3 windows
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=256, kernel_size=3)
        self.pool1 = nn.MaxPool2d(2,2) # Non learning layer, takes 2x2 windows and takes it to get the max value of them

        self.conv2 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3)
        self.pool2 = nn.MaxPool2d(2,2)

        # Tiny image
        self.conv3 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=2)
        self.pool3 = nn.MaxPool2d(2,2)

        # That's enough convultions!
        # Get thy vector
        self.flatten = nn.Flatten()

        # lINEAR, where we now compress the vector
        # self.fc1 = nn.Linear(in_features=4096, out_features=1024)
        self.fc1 = nn.Linear(in_features=200704, out_features=1024)
        # Helps with overfitting by letting us drop values randomly
        self.drop1 = nn.Dropout(p=.3)

        self.fc2 = nn.Linear(in_features=1024, out_features=1024)
        self.drop2 = nn.Dropout(p=.3)

        things = 2 # Our labels
        # Collapses our great filter into a values
        self.out = nn.Linear(in_features=1024, out_features=things)



    """
    The magic of my neural net of doom.
    """
    def forward(self, x):
        # Runs through the conv layer
        x = F.relu(self.conv1(x)) # Non linear function, gets features?
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x)) 
        x = self.pool3(x)
        x = self.flatten(x)
        # x = F.relu(self.fc1(x))
        # x = self.drop1(x)
        # x = F.relu(self.fc2(x))
        # x = self.drop2(x)
        # x = self.out(x)
        # x = self.pool(F.relu(self.conv1(x)))
        # x = self.pool(F.relu(self.conv2(x)))
        # x = torch.flatten(x, 1) # flatten all dimensions except batch
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = self.fc3(x)
        return x