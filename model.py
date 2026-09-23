import torch
import torch.nn as nn


class HandwritingCNN(nn.Module):

    def __init__(self):
        super().__init__()

        # First convolution:
        # [batch, 1, 28, 28]
        # ->
        # [batch, 16, 28, 28]
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=16,
            kernel_size=3,
            padding=1
        )

        # Second convolution:
        # [batch, 16, 14, 14]
        # ->
        # [batch, 32, 14, 14]
        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            padding=1
        )

        self.relu = nn.ReLU()

        # Halves width and height
        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        # After two pooling operations:
        # 28 -> 14 -> 7
        #
        # 32 * 7 * 7 = 1568
        self.fc1 = nn.Linear(
            32 * 7 * 7,
            128
        )

        # Ten possible digits
        self.fc2 = nn.Linear(
            128,
            10
        )


    def forward(self, x):

        # Convolution block 1
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)

        # Convolution block 2
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)

        # [batch, 32, 7, 7]
        # ->
        # [batch, 1568]
        x = torch.flatten(x, 1)

        # Fully connected layers
        x = self.fc1(x)
        x = self.relu(x)

        # Ten logits
        x = self.fc2(x)

        return x