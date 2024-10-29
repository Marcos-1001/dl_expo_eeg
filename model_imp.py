import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Define the CNN model for 2D input (channels, samples)
class EEG_CNN(nn.Module):
    def __init__(self):
        super(EEG_CNN, self).__init__()
        
        # First Convolutional Layer
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=32, kernel_size=10, stride=1, padding=0)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Second Convolutional Layer
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=10, stride=1, padding=0)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Third Convolutional Layer
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=10, stride=1, padding=0)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 47, 128)  # Adjust based on flattened size
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 11)  # Output layer for 11 classes
        
        # Dropout layer
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x)
        # Convolutional and pooling layers
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = self.pool3(torch.relu(self.conv3(x)))
        
        # Flatten the output for fully connected layers
        # get x size
        x = x.view(-1, 128 * 47)


        # Fully connected layers
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.dropout(torch.relu(self.fc2(x)))
        x = self.fc3(x)  # No activation at the output layer, handled by loss function
        
        return x
