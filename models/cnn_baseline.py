import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN_baseline(nn.Module):
    def __init__(self):
        """
        Baseline CNN model (without dropout, for testing two non-MC-dropout-based acquisition functions.)
        """
        super().__init__()

        # ---- Convolutional layers ----
        self.conv1 = nn.Conv2d(1, 32, kernel_size=4)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=4)
        self.pool = nn.MaxPool2d(2)

        # ---- Fully connected layers ----
        self.fc1 = nn.Linear(32 * 11 * 11, 128)   # after convs and pool, get height and length 11.
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        """
        Forward pass through the CNN_baseline model
        Args:
            x: input tensor of shape [B, 1, 28, 28]
        Returns:
            logits: output tensor of shape [B, 10]
        """

        # x: [B, 1, 28, 28]
        x = F.relu(self.conv1(x))      # -> [B, 32, 25, 25]
        x = F.relu(self.conv2(x))      # -> [B, 32, 22, 22]
        x = self.pool(x)               # -> [B, 32, 11, 11]

        x = x.view(x.size(0), -1)      # -> [B, 32, 11, 11]
        feat = F.relu(self.fc1(x))     # -> [B, 128]

        out = self.fc2(feat)           # -> [B, 10]
        return out

    def get_feature(self, x):
        """
        Obtain features from the trained model instance as encoder.
        Args:
            x: input tensor of shape [B, 1, 28, 28]
        Returns:
            logits: output tensor of shape [B, 10]
        """

        # x: [B, 1, 28, 28]
        x = F.relu(self.conv1(x))      # -> [B, 32, 25, 25]
        x = F.relu(self.conv2(x))      # -> [B, 32, 22, 22]
        x = self.pool(x)               # -> [B, 32, 11, 11]

        x = x.view(x.size(0), -1)      # -> [B, 32, 11, 11]
        feat = F.relu(self.fc1(x))     # -> [B, 128]

        return feat
