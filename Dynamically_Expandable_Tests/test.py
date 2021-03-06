
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
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

# Hyperparameters
n_epochs = 100
batch_size = 128
learning_rate = 0.01
print('Number of Epochs: ', 7 * n_epochs, 
      '\nBatch Size: ', batch_size,
      '\nLearning Rate: ', learning_rate)


# Train base line network (starting at max size)
base_net = DynaNet(n_epochs= 1 * n_epochs,
                   num_kernels_layer2 = 18,
                   batch_size = batch_size,
                   learning_rate = learning_rate)


# Train dynamic network
dyna_net = DynaNet(n_epochs= n_epochs,
                   num_kernels_layer1 = 2,
                   num_kernels_layer2 = 6,
                   num_kernels_layer3 = 40,
                   batch_size = batch_size,
                   learning_rate = learning_rate,
                   freeze_train_ratio = 0.8)

for i in range(1):

    if i in [0,1]:
        dyna_net.expand(added_kernels_layer3 = 40)

    if i in [2,3]:
        dyna_net.expand(added_kernels_layer2 = 6)

    if i in [4,5]:
        dyna_net.expand(added_kernels_layer1 = 2)

print("\n===============================")
print("Baseline Results ")
print("-------------------------------")
print("Test Accuracy: ",base_net.acc)

print("\nDynamic Results ")
print("-------------------------------")
print("Test Accuracy: ",dyna_net.acc)
print("===============================")

print("\n--- The experiment took %s seconds to run ---" % (round((time.time() - start_time))))
print("---                     %s minutes to run ---" % (round(((time.time() - start_time)/60))))
print("---                     %s hours to run ---" % (round(((time.time() - start_time)/3600))))


