
'''
This script is just used for testing code blocks
'''

import numpy as np
import torch
from Dynamic_Network import DynaNet


# Train base line network (starting at max size)
base_net = DynaNet(n_epochs= 10,
                   num_kernels_layer2 = 18)


# Train dynamic network
dyna_net = DynaNet(n_epochs=10,
                   num_kernels_layer1 = 2,
                   num_kernels_layer2 = 6,
                   num_kernels_layer3 = 40)

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

print("===============================")
print("Baseline Results ")
print("-------------------------------")
print("Validation Loss:     ", base_net.val_loss)
print("Validation Accuracy: ",base_net.val_acc)

print("\nDynamic Results ")
print("-------------------------------")
print("Validation Loss:     ", dyna_net.val_loss)
print("Validation Accuracy: ",dyna_net.val_acc)
print("===============================")



