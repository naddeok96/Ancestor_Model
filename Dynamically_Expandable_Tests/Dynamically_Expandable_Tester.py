# Imports
import torch
from torch import nn
import torchvision
import numpy as np
import torch.nn.functional as F
from torchsummary import summary
from Adjustable_LeNet import AdjLeNet
from CIFARsetup import CIFAR10_Setup
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import time
import matplotlib.pyplot as plt
from silence import shh

# Test Parameters
num_experiments = 10
num_epochs = 10

# Number of Possible Kernels in each Conv Layer
num_kernels_layer1_set = [3, 6, 9]
num_kernels_layer2_set = [8, 16, 24]
num_kernels_layer3_set = [60, 120, 180]

# Initialize Variables
val_losses_ascend_total = np.zeros(9)
val_accuracies_ascend_total = np.zeros(9)
val_losses_descend_total = np.zeros(9)
val_accuracies_descend_total = np.zeros(9)


# Outer Loop is for Each Experiment
for k in range(num_experiments):

    # Reset values for every new experiment
    val_losses_ascend = []
    val_accuracies_ascend = []
    comb_ascend = []

    val_losses_descend = []
    val_accuracies_descend = []
    comb_descend = []

    # Walk through each possible combination of kernels
    for i in range(9):

        # Change Kernels based off iteration
        if i <= 2:
            num_kernels_layer1_ascend = num_kernels_layer1_set[i]
            num_kernels_layer3_descend = num_kernels_layer3_set[i]

        num_kernels_layer2_ascend = num_kernels_layer2_set[0]
        num_kernels_layer3_ascend = num_kernels_layer3_set[0]

        num_kernels_layer1_descend = num_kernels_layer1_set[0]
        num_kernels_layer2_descend = num_kernels_layer2_set[0]
        
        if i > 2 and i <= 5:
            num_kernels_layer2_ascend = num_kernels_layer2_set[i-3]
            num_kernels_layer2_descend = num_kernels_layer2_set[i-3]
        
        if i > 5:
            num_kernels_layer3_ascend = num_kernels_layer3_set[i-5]
            num_kernels_layer1_descend = num_kernels_layer1_set[i-5]


        # Initalize the LeNets w/ number of kernels
        CNNascend = DLeNet(num_kernels_layer1 = num_kernels_layer1_ascend,
                           num_kernels_layer2 = num_kernels_layer2_ascend,
                           num_kernels_layer3 = num_kernels_layer3_ascend)

        CNNdescend = DLeNet(num_kernels_layer1 = num_kernels_layer1_descend,
                            num_kernels_layer2 = num_kernels_layer2_descend,
                            num_kernels_layer3 = num_kernels_layer3_descend)


        with shh(): # shh will prevent the print out of .trainNet

            # These will load the CIFAR-10 dataset and train the LeNets 
            val_loss_ascend, val_acc_ascend = training(CNNascend).trainNet(batch_size = 128,
                                                                            n_epochs = num_epochs,
                                                                            learning_rate=0.001)

            val_loss_descend, val_acc_descend = training(CNNdescend).trainNet(batch_size = 128,
                                                                            n_epochs = num_epochs,
                                                                            learning_rate=0.001)

        # These are list of metrics from each architerture
        val_losses_ascend.append(val_loss_ascend)
        val_accuracies_ascend.append(val_acc_ascend)
        comb_ascend.append([num_kernels_layer1_ascend,
                            num_kernels_layer2_ascend,
                            num_kernels_layer3_ascend])

        val_losses_descend.append(val_loss_descend)
        val_accuracies_descend.append(val_acc_descend)
        comb_descend.append([num_kernels_layer1_descend,
                            num_kernels_layer2_descend,
                            num_kernels_layer3_descend])

    # These are the running totals of all the experiments
    val_losses_ascend_total += np.asarray(val_losses_ascend)
    val_accuracies_ascend_total += np.asarray(val_accuracies_ascend)
    val_losses_descend_total += np.asarray(val_losses_descend)
    val_accuracies_descend_total += np.asarray(val_accuracies_descend)   
    
# This will print out the results of the experiments
print("Number of Experiments:\n",num_experiments,
      "\nCombinations Kernels for Ascending:\n",comb_ascend,
      "\nCombinations Kernels for Descending:\n",comb_descend,
      "\nAvg Val Loss Ascending:\n", val_losses_ascend_total/num_experiments,
      "\nAvg Val Acc Ascending:\n", val_accuracies_ascend_total/num_experiments,
      "\nAvg Val Loss Descending:\n", val_losses_descend_total/num_experiments,
      "\nAvg Val Acc Descending:\n", val_accuracies_descend_total/num_experiments)

