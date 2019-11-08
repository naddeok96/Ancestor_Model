
'''
This script is just used for testing code blocks
'''

import numpy as np
import torch
from Dynamic_Network import DynaNet

'''
# Dynamically Expand
LeNet1 = DLeNet(num_kernels_layer1=1)

print("\nOrginal LeNet1 Conv1:   \n", LeNet1.conv1.weight.data)

training(LeNet1).trainNet(batch_size = 128,
                          n_epochs = 5,
                          learning_rate = 0.01)

print("\nTrained LeNet1 Conv1:   \n", LeNet1.conv1.weight.data)

LeNet2 = DLeNet(num_kernels_layer1=2)

print("\nOrginal LeNet2 Conv1:   \n", LeNet2.conv1.weight.data)

LeNet2.conv1.weight.data[0,0,:] = LeNet1.conv1.weight.data[0,0,:]

print("\nModified LeNet2 Conv1:   \n", LeNet2.conv1.weight.data)
'''

net = DynaNet(n_epochs=1, num_kernels_layer1=1)

print(net.net.conv1.weight.data)

net.expand(added_kernels_layer1 = 1,
           added_kernels_layer2 = 2,
           added_kernels_layer3 = 4)

print(net.net.conv1.weight.data)




