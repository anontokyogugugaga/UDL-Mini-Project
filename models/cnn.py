import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, dropout_conv=0.25, dropout_fc=0.5):
        """
        CNN model
        Args:
            dropout_conv: dropout rate for convolutional layers
            dropout_fc: dropout rate for fully connected layers
        """
        super().__init__()

        # ---- Convolutional layers ----
        self.conv1 = nn.Conv2d(1, 32, kernel_size=4)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=4)
        self.pool = nn.MaxPool2d(2)

        self.dropout_conv = nn.Dropout(p=dropout_conv)

        # ---- Fully connected layers ----
        self.fc1 = nn.Linear(32 * 11 * 11, 128)   # after convs and pool, get height and length 11.
        self.dropout_fc = nn.Dropout(p=dropout_fc)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        """
        Forward pass through the CNN model
        Args:
            x: input tensor of shape [B, 1, 28, 28]
        Returns:
            logits: output tensor of shape [B, 10]
        """

        # x: [B, 1, 28, 28]
        x = F.relu(self.conv1(x))      # -> [B, 32, 25, 25]
        x = F.relu(self.conv2(x))      # -> [B, 32, 22, 22]
        x = self.pool(x)               # -> [B, 32, 11, 11]
        x = self.dropout_conv(x)

        x = x.view(x.size(0), -1)      # -> [B, 32, 11, 11]
        x = F.relu(self.fc1(x))
        x = self.dropout_fc(x)

        logits = self.fc2(x)           # -> [B, 10]
        return logits



