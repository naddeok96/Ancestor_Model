
import torch
from torch import nn
import torchvision
import numpy as np
import torch.nn.functional as F
from torchsummary import summary
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import time

class CIFAR10_Setup:

    def __init__(self,net,
                      n_train_samples = 20000,
                      n_val_samples = 10000,
                      n_test_samples = 10000):
        
        super(CIFAR10_Setup,self).__init__()

        self.net = net.cuda()
        self.n_train_samples = n_train_samples
        self.n_val_samples = n_val_samples
        self.n_test_samples = n_test_samples


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

        self.train_sampler = SubsetRandomSampler(np.arange(self.n_train_samples,dtype=np.int64))
        self.val_sampler = SubsetRandomSampler(np.arange(self.n_train_samples, self.n_train_samples + self.n_val_samples,dtype=np.int64))
        self.test_sampler = SubsetRandomSampler(np.arange(self.n_test_samples, dtype=np.int64))

        #Test and validation loaders have constant batch sizes, so we can define them directly
        self.test_loader = torch.utils.data.DataLoader(self.test_set,
                                                batch_size=4,
                                                sampler=self.test_sampler,
                                                num_workers=2)
                                                
        self.val_loader = torch.utils.data.DataLoader(self.train_set,
                                                batch_size = 128,
                                                sampler=self.val_sampler,
                                                num_workers=2)


    # Fucntion to break training set into batches
    def get_train_loader(self,batch_size):
        train_loader = torch.utils.data.DataLoader(self.train_set,
                                                   batch_size = batch_size,
                                                   sampler = self.train_sampler,
                                                   num_workers = 2)
        return(train_loader)

    # Define our loss function and optimazation to use
    def createLossAndOptimizer(self,net, lr=0.01):
        loss = torch.nn.CrossEntropyLoss()
        optimzer = torch.optim.SGD(net.parameters(), lr=lr, momentum= 0.9, weight_decay=0.0001)     

        return(loss,optimzer) 

    # Function to train the network     
    def fit_model(self, batch_size, n_epochs, learning_rate, freeze_name = 'conv1', freeze_param = '[0:1]'):
        
        #Get training data
        self.train_loader = self.get_train_loader(batch_size)
        n_batches = len(self.train_loader)

        #Create our loss and optimizer functions
        loss, optimizer = self.createLossAndOptimizer(self.net, learning_rate)
        
        #Time for printing
        training_start_time = time.time()


    #Loop for n_epochs
        for epoch in range(n_epochs):
            
            running_loss = 0.0
            print_every = n_batches // 10
            start_time = time.time()
            total_train_loss = 0
            
            for i, data in enumerate(self.train_loader, 0):
                
                #Reset the train loader and apply a counter
                inputs, labels = data

                # Push input to gpus
                inputs, labels = inputs.cuda(), labels.cuda()

                #Set the parameter gradients to zero
                optimizer.zero_grad()
                
                #Forward pass, backward pass, optimize
                outputs = self.net(inputs) # Forward pass

                soft_outputs = F.softmax(outputs, dim=1)
                predicted = torch.max(soft_outputs, dim = 1)[1]
                acc = torch.sum(torch.eq(predicted, labels)).item()/predicted.nelement()


                loss_size = loss(outputs, labels) # calculate loss

                


                loss_size.backward() # Find the gradient for each parameter

                
                for name, param in self.net.named_parameters():
                    print(self.net.[name].grad)

                '''
                print(self.net.conv1.weight.grad)
                self.net.['freeze_name'].weight.grad[freeze_param, :, :] = 0
                print(self.net.conv1.weight.grad)
                '''
                
                print("Stop Here \n")
                exit()
                optimizer.step() # Parameter update
                
                #Print statistics
                running_loss += loss_size.item()
                total_train_loss += loss_size.item()
                

            #At the end of the epoch, do a pass on the validation set
            total_val_loss = 0
            for inputs, labels in self.val_loader:
                
                # Push input to gpu
                inputs, labels = inputs.cuda(), labels.cuda()

                #Forward pass
                val_outputs = self.net(inputs)

                val_soft_outputs = F.softmax(val_outputs, dim=1)
                predicted = torch.max(val_soft_outputs, dim = 1)[1]
                
                val_acc = torch.sum(torch.eq(predicted, labels)).item()/predicted.nelement()
                val_loss_size = loss(val_outputs, labels)
                total_val_loss += val_loss_size.item()
            

        return (total_val_loss / len(self.val_loader), val_acc)
