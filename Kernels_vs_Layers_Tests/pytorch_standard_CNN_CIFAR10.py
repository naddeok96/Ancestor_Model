
# Imports
import numpy as np 
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.autograd import Variable
import torch.nn.functional as F
import time
'''
This is a walkthrough found at https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
'''


# Set seed for reproducability
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

# Pull in data
transform = transforms.Compose([transforms.ToTensor(), # Images are of size (3,32,32)
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) # this will allow us to convert the images into tensors and normalize about 0.5

train_set = torchvision.datasets.CIFAR10(root='./cifardata',
                                        train=True,
                                        download=True,
                                        transform=transform)

test_set = torchvision.datasets.CIFAR10(root='./cifardata',
                                        train=False,
                                        download=True,
                                        transform=transform)


# Break data into train, test and validation
n_train_samples = 20000
n_val_samples = 5000
n_test_samples = 5000

train_sampler = SubsetRandomSampler(np.arange(n_train_samples,dtype=np.int64))
val_sampler = SubsetRandomSampler(np.arange(n_train_samples, n_train_samples + n_val_samples,dtype=np.int64))
test_sampler = SubsetRandomSampler(np.arange(n_test_samples, dtype=np.int64))

# Design a CNN using Pytorch
class CNN(torch.nn.Module):

     # Shape for input x is (3, 32, 32)

    def __init__(self): # Things to be done when intialized
        super(CNN, self).__init__() # Inherent all the torch.nn.Module properties

        # Input to first conv layer
        self.conv1 = torch.nn.Conv2d(3, # Number of inputs (RGB)
                                    18, # Number of Kernels used
                                    kernel_size=3, # Size of Kernels
                                    stride=1, # Strides of the Kernels
                                    padding=1) # Make the output image equal to the input in shape

        self.pool = torch.nn.MaxPool2d(kernel_size=2,
                                        stride=2,
                                        padding=0)

        # There are 18 channels and the w and h are equal
        # W or H can be calculated by (input - kernel_size + (2*padding))/stride + 1
        # Therefore W = H = (32 - 2 + (2*0))/2 + 1 = 16
        # The next input size then is 18*16*16 = 4608
        # The output was arbitarily choosen to be 64
        self.fc1 = torch.nn.Linear(4608,64)

        # The output must be equal to the number of classes
        self.fc2 = torch.nn.Linear(64,10)

    # Function to perform a forward pass of the network
    def forward(self, x):

        # Compute the activation of the first layer
        x = F.relu(self.conv1(x))

        # Take the max pool of this
        x = self.pool(x)

        # Flatten to be accepted to the fully connected layer
        x = x.view(-1, 4608) # the 4608 is one dimension the -1 will figure out what the other dimension should be

        # Now compute the activation of the first fully connected layer
        x = F.relu(self.fc1(x))

        # Now compute the last layer with no activation which will be done later
        x = self.fc2(x)

        return(x)

# Fucntion to break training set into batches
def get_train_loader(batch_size):
    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size = batch_size,
                                               sampler = train_sampler,
                                               num_workers = 2)
    return(train_loader)

#Test and validation loaders have constant batch sizes, so we can define them directly
test_loader = torch.utils.data.DataLoader(test_set,
                                         batch_size=4,
                                         sampler=test_sampler,
                                         num_workers=2)
val_loader = torch.utils.data.DataLoader(train_set,
                                         batch_size=128,
                                         sampler=val_sampler,
                                         num_workers=2)

# Define our loss function and optimazation to use
def createLossAndOptimizer(net, lr=0.001):
    loss = torch.nn.CrossEntropyLoss()
    optimzer = torch.optim.Adam(net.parameters(), lr=lr)     

    return(loss,optimzer)

# Function to train the network     
def trainNet(net, batch_size, n_epochs, learning_rate):
    #Print all of the hyperparameters of the training iteration:
    print("===== HYPERPARAMETERS =====")
    print("batch_size=", batch_size)
    print("epochs=", n_epochs)
    print("learning_rate=", learning_rate)
    print("=" * 30)
    
    #Get training data
    train_loader = get_train_loader(batch_size)
    n_batches = len(train_loader)

    #Create our loss and optimizer functions
    loss, optimizer = createLossAndOptimizer(net, learning_rate)
    
    #Time for printing
    training_start_time = time.time()


   #Loop for n_epochs
    for epoch in range(n_epochs):
        
        running_loss = 0.0
        print_every = n_batches // 10
        start_time = time.time()
        total_train_loss = 0
        
        for i, data in enumerate(train_loader, 0):
            
            #Reset the train loader and apply a counter
            inputs, labels = data

            #Set the parameter gradients to zero
            optimizer.zero_grad()
            
            #Forward pass, backward pass, optimize
            outputs = net(inputs) # Forward pass
            loss_size = loss(outputs, labels) # calculate loss
            loss_size.backward() # Find the gradient for each parameter
            optimizer.step() # Parameter update
            
            #Print statistics
            running_loss += loss_size.item()
            total_train_loss += loss_size.item()
            
            #Print every 10th batch of an epoch
            if (i + 1) % (print_every + 1) == 0:
                print("Epoch {}, {:d}% \t train_loss: {:.2f} took: {:.2f}s".format(
                        epoch+1, int(100 * (i+1) / n_batches), running_loss / print_every, time.time() - start_time))
                #Reset running loss and time
                running_loss = 0.0
                start_time = time.time()
            
        #At the end of the epoch, do a pass on the validation set
        total_val_loss = 0
        for inputs, labels in val_loader:
            
            #Forward pass
            val_outputs = net(inputs)
            val_loss_size = loss(val_outputs, labels)
            total_val_loss += val_loss_size.item()
            
        print("Validation loss = {:.2f}".format(total_val_loss / len(val_loader)))
        
    print("Training finished, took {:.2f}s".format(time.time() - training_start_time))


# Lets Run the Fucking Code
CNN = CNN()
trainNet(CNN, 
        batch_size = 256,
        n_epochs = 1,
        learning_rate=0.001)


        



