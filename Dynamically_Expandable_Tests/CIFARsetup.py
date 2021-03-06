
import torch
from torch import nn
import torchvision
import numpy as np
import torch.nn.functional as F
from torchsummary import summary
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import time
import operator

class CIFAR10_Setup:

    def __init__(self,net):
        
        super(CIFAR10_Setup,self).__init__()

        self.net = net.cuda()

        # Pull in data
        self.transform = transforms.Compose([transforms.ToTensor(), # Images are of size (3,32,32)
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) # this will allow us to convert the images into tensors and normalize about 0.5

        self.train_set = torchvision.datasets.CIFAR10(root='./cifardata',
                                                train=True,
                                                download=True,
                                                transform=self.transform)

        self.test_set = torchvision.datasets.CIFAR10(root='./cifardata',
                                                train=False,
                                                download=True,
                                                transform=self.transform)

        #Test and validation loaders have constant batch sizes, so we can define them directly
        self.test_loader = torch.utils.data.DataLoader(self.test_set,
                                                       batch_size=100,
                                                       shuffle=False,
                                                       num_workers=8,
                                                       pin_memory=True)

    # Fucntion to break training set into batches
    def get_train_loader(self,batch_size):
        train_loader = torch.utils.data.DataLoader(self.train_set,
                                                   batch_size = batch_size,
                                                   shuffle = True,
                                                   num_workers = 8,
                                                   pin_memory=True)
        return(train_loader)

    # Function to train the network     
    def fit_model(self, batch_size, n_epochs, learning_rate, freeze_name = [], freeze_param = []):
        
        #Get training data
        self.train_loader = self.get_train_loader(batch_size)
        n_batches = len(self.train_loader)

        #Create our loss and optimizer functions
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.net.parameters(), lr=0.01, momentum= 0.9, weight_decay=0.0001) 

        #Loop for n_epochs
        for epoch in range(n_epochs):            
            for i, data in enumerate(self.train_loader, 0):
                
                #Reset the train loader and apply a counter
                inputs, labels = data

                # Push input to gpus
                inputs, labels = inputs.cuda(), labels.cuda()

                #Set the parameter gradients to zero
                optimizer.zero_grad()
                
                #Forward pass, backward pass, optimize
                outputs = self.net(inputs) # Forward pass
                loss = criterion(outputs, labels) # calculate loss
                loss.backward() # Find the gradient for each parameter

                for name, params in zip(freeze_name, freeze_param):
                    new_grads = operator.attrgetter(name + '.grad')(self.net)[params, :, :]
                    optimizer.zero_grad()
                    operator.attrgetter(name + '.grad')(self.net)[params, :, :] = new_grads

                optimizer.step() # Parameter update
                

        # At the end of training run a test
        total_tested = 0
        correct = 0
        for inputs, labels in self.test_loader:
            
            # Push input to gpu
            inputs, labels = inputs.cuda(), labels.cuda()

            #Forward pass
            outputs = self.net(inputs)
            _, predicted = torch.max(outputs.data, 1)

            total_tested += labels.size(0)
            correct += (predicted == labels).sum().item()
        return (correct/total_tested)
