"""A minimal convolutional network used for small-scale experiments.

This file defines `BasicCNN`, a tiny CNN with two conv layers and two
fully-connected layers. It is mainly useful for tests and quick checks.
"""

import torch


class BasicCNN(torch.nn.Module):
    """Simple CNN: conv -> pool -> conv -> pool -> FC -> FC.

    Args:
        in_channels: input channel count (default 1)
        out_channels: output classes (default 10)
    """
    def __init__(self, in_channels=1, out_channels=10):
        super(BasicCNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.fc1 = torch.nn.Linear(32 * 56 * 56, 128)
        self.fc2 = torch.nn.Linear(128, out_channels)

    def forward(self, x):
        """Forward pass returning logits of shape `[B, out_channels]`."""
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 32 * 56 * 56)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x