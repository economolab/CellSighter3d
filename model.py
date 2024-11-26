from torchvision.models.video import r3d_18
import torch
from torch import nn


class Model(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(Model, self).__init__()
        # Load a pretrained 3D ResNet (ResNet-18 for 3D data as an example)
        self.model = r3d_18()
        
        # Modify the first convolutional layer to match input channels
        self.model.stem[0] = nn.Conv3d(
            input_channels, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False
        )

        # Adjust the final fully connected layer to output the desired number of classes
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

        # Initialize weights for the modified convolutional layer
        nn.init.kaiming_normal_(self.model.stem[0].weight, mode="fan_out", nonlinearity="relu")

        # Add a softmax layer for evaluation
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.model(x)
        if not self.training:
            x = self.softmax(x)
        return x
