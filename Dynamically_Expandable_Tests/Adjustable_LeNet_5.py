'''
This model allows for a LeNet to be generated with explicit kernels in each conv layer
'''

# Imports
import torch
from torch import nn
import torch.nn.functional as F
from torchsummary import summary

class AdjLeNet(nn.Module):

    def __init__(self, num_classes = 10,
                       num_kernels_layer1 = 6, 
                       num_kernels_layer2 = 16, 
                       num_kernels_layer3 = 120):

        super(DLeNet,self).__init__()
        
        self.num_classes = num_classes

        self.num_kernels_layer1 = num_kernels_layer1
        self.num_kernels_layer2 = num_kernels_layer2
        self.num_kernels_layer3 = num_kernels_layer3

        # Input (3,32,32)
        # Layer 1
        self.conv1 = nn.Conv2d(3, # Input channels
                               self.num_kernels_layer1, # Output Channel 
                               kernel_size = 5, 
                               stride = 1, 
                               padding = 0) # Output = (3,28,28)

        # Layer 2
        self.pool1 = nn.MaxPool2d(kernel_size = 2, 
                                  stride = 2, 
                                  padding = 0) # Output = (num_kernels_layer1,14,14)

        # Layer 3
        self.conv2 = nn.Conv2d(self.num_kernels_layer1,
                               self.num_kernels_layer2,
                               kernel_size = 5, 
                               stride = 1, 
                               padding = 0) # Output = (num_kernels_layer2,10,10)

        # Layer 4
        self.pool2 = nn.MaxPool2d(kernel_size = 2, 
                                  stride = 2, 
                                  padding = 0) # Output = (num_kernels_layer3,5,5)

        # Layer 5
        self.conv3 = nn.Conv2d(self.num_kernels_layer2,
                               self.num_kernels_layer3,
                               kernel_size = 5, 
                               stride = 1, 
                               padding = 0) # Output = (num_kernels_layer4,1,1)

        self.fc1 = nn.Linear(self.num_kernels_layer3, 84)

        self.fc2 = nn.Linear(84, self.num_classes)
        

    def forward(self, x):
        
        x = torch.tanh(self.conv1(x))
        x = self.pool1(x)
        x = torch.tanh(self.conv2(x))
        x = self.pool2(x)
        x = torch.tanh(self.conv3(x))
        x = x.view(-1, self.num_kernels_layer3 * 1 * 1)
        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)

        return (x)


