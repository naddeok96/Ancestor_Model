'''
This will be the base skeleton of CNNs used to grow
for CIFAR10
'''
# Imports
import torch
from torch import nn
import torch.nn.functional as F
from torchsummary import summary

class CNN(nn.Module):

    def __init__(self, num_layers = 1, num_classes = 10,
                       num_kernels_layer1 = 18, num_kernels_layer2 = 27, num_kernels_layer3 = 36):

        super(CNN,self).__init__()

        self.num_layers = num_layers
        self.num_classes = num_classes

        self.num_kernels_layer1 = num_kernels_layer1
        self.num_kernels_layer2 = num_kernels_layer2
        self.num_kernels_layer3 = num_kernels_layer3

        # Input (3,32,32)
        self.conv1 = nn.Conv2d(3, # Input channels
                               self.num_kernels_layer1,  
                               kernel_size = 3, 
                               stride = 1, 
                               padding = 1)
        self.pool1 = nn.MaxPool2d(kernel_size = 2, 
                                  stride = 2, 
                                  padding = 0) # Output = (num_kernels_layer1,16,16)

        self.conv2 = nn.Conv2d(self.num_kernels_layer1,
                               self.num_kernels_layer2,
                               kernel_size = 3, 
                               stride = 1, 
                               padding = 1)
        self.pool2 = nn.MaxPool2d(kernel_size = 2, 
                                  stride = 2, 
                                  padding = 1) # Output = (num_kernels_layer2,9,9)

        self.conv3 = nn.Conv2d(self.num_kernels_layer2,
                               self.num_kernels_layer3, 
                               kernel_size = 3, 
                               stride = 1, 
                               padding = 1)
        self.pool3 = nn.MaxPool2d(kernel_size = 3, 
                                  stride = 3, 
                                  padding =  0) # Output = (num_kernels_layer3,3,3)

        self.fc1 = nn.Linear(self.fc1_size_func(self.num_layers, 
                                                self.num_kernels_layer1,
                                                self.num_kernels_layer2,
                                                self.num_kernels_layer3),64)

        self.fc2 = nn.Linear(64, self.num_classes)
        

    def forward(self, x):
        
        if self.num_layers == 1:
            x = F.relu(self.conv1(x))
            x = self.pool1(x)
            x = x.view(-1, self.num_kernels_layer1 * 16 * 16)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)

        if self.num_layers == 2:
            x = F.relu(self.conv1(x))
            x = self.pool1(x)
            x = F.relu(self.conv2(x))
            x = self.pool2(x)
            x = x.view(-1, self.num_kernels_layer2 * 9 * 9)
            x = F.relu(self.fc1(x))
            x= self.fc2(x)

        if self.num_layers == 3:
            x = F.relu(self.conv1(x))
            x = self.pool1(x)
            x = F.relu(self.conv2(x))
            x = self.pool2(x)
            x = F.relu(self.conv3(x))
            x = self.pool3(x)
            x = x.view(-1, self.num_kernels_layer3 * 3 * 3)
            x = F.relu(self.fc1(x))
            x= self.fc2(x)

        return (x)
      
    def fc1_size_func(self,num_layers, num_kernels_layer1, num_kernels_layer2, num_kernels_layer3):
        if num_layers == 1:
            fc1_size = num_kernels_layer1 * 16 * 16
        if num_layers == 2:
            fc1_size = num_kernels_layer2 * 9 * 9
        if num_layers == 3:
            fc1_size = num_kernels_layer3 * 3 * 3
        return int(fc1_size)

