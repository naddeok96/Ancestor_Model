
'''
This script is just used for testing code blocks
'''

import numpy as np
import torch
from Dynamic_Network import DynaNet
import time
import os

# Track run time
start_time = time.time()

# GPU Setup
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# For a single device (GPU 2)
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

# Hyperparameters
n_epochs = 100
batch_size = 128
learning_rate = 0.001
print('Number of Epochs: ', n_epochs, 
      '\nBatch Size: ', batch_size,
      '\nLearning Rate: ', learning_rate)

# Train base line network (starting at max size)
base_net = DynaNet(n_epochs= n_epochs,
                   num_kernels_layer2 = 18,
                   batch_size = batch_size,
                   learning_rate = learning_rate)


# Train dynamic network
dyna_net = DynaNet(n_epochs= n_epochs,
                   num_kernels_layer1 = 2,
                   num_kernels_layer2 = 6,
                   num_kernels_layer3 = 40,
                   batch_size = batch_size,
                   learning_rate = learning_rate)

for i in range(6):

    if i in [0,1]:
        dyna_net.expand(added_kernels_layer3 = 40)
        dyna_net.train()

    if i in [2,3]:
        dyna_net.expand(added_kernels_layer2 = 6)
        dyna_net.train()

    if i in [4,5]:
        dyna_net.expand(added_kernels_layer1 = 2)
        dyna_net.train()

print("\n===============================")
print("Baseline Results ")
print("-------------------------------")
print("Validation Loss:     ", base_net.val_loss)
print("Validation Accuracy: ",base_net.val_acc)

print("\nDynamic Results ")
print("-------------------------------")
print("Validation Loss:     ", dyna_net.val_loss)
print("Validation Accuracy: ",dyna_net.val_acc)
print("===============================")

print("--- The experiemtn took %s seconds to run ---" % (time.time() - start_time))


