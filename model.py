from torchvision.models.video import r2plus1d_50
import torch
from torch import nn


class Model(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(Model, self).__init__()
        
        # Load the pretrained R(2+1)D ResNet-50 model
        self.model = r2plus1d_50(pretrained=True)
        
        # Replace the first convolutional layer to match the input channels
        self.model.stem[0] = nn.Conv3d(
            in_channels=input_channels,  # E.g., 1 for grayscale, 3 for RGB
            out_channels=45,  # Matches original R(2+1)D output channels
            kernel_size=(3, 7, 7),  # Default spatiotemporal kernel
            stride=(1, 2, 2),
            padding=(1, 3, 3),
            bias=False
        )
        
        # Initialize the weights of the new conv layer
        nn.init.kaiming_normal_(self.model.stem[0].weight, mode='fan_out', nonlinearity='relu')
        
        # Replace the final fully connected layer to match the number of classes
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        
        # Initialize the weights of the new fully connected layer
        nn.init.kaiming_normal_(self.model.fc.weight, mode='fan_out', nonlinearity='relu')
        
        # Softmax for evaluation mode
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.model(x)
        if not self.training:
            x = self.softmax(x)
        return x