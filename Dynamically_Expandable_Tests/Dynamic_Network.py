
# Import
from Adjustable_LeNet_5 import AdjLeNet
from CIFARsetup import CIFAR10_Setup
from silence import shh
from torchsummary import summary

class DynaNet:

    def __init__(self, num_classes = 10,
                       num_kernels_layer1 = 6, 
                       num_kernels_layer2 = 16, 
                       num_kernels_layer3 = 120,
                       batch_size = 128,
                       n_epochs = 10,
                       learning_rate = 0.01):

        super(DynaNet,self).__init__()

        self.num_classes = num_classes
        self.num_kernels_layer1 = num_kernels_layer1
        self.num_kernels_layer2 = num_kernels_layer2 
        self.num_kernels_layer3 = num_kernels_layer3
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate

        print("\nModel Setup:  ")
        self.net = AdjLeNet(num_classes= self.num_classes,
                            num_kernels_layer1 = self.num_kernels_layer1,
                            num_kernels_layer2 = self.num_kernels_layer2,
                            num_kernels_layer3 = self.num_kernels_layer3)

        summary(self.net, input_size=(3, 32, 32), device="cpu") # Summarize the model

        self.train()

    def train(self):
        # These will load the CIFAR-10 dataset and train the LeNets 
        print("\n\nCIFAR-10 Training:")
        print("----------------------------------------------------------------")
        self.val_loss, self.val_acc = CIFAR10_Setup(self.net).train(batch_size = self.batch_size,
                                                                    n_epochs = self.n_epochs,
                                                                    learning_rate= self.learning_rate)
        print("\nValidation Loss:    ",self.val_loss)
        print("Validation Accuracy:", self.val_acc)
        print("----------------------------------------------------------------")

    def expand(self, added_kernels_layer1 = 0,
                     added_kernels_layer2 = 0,
                     added_kernels_layer3 = 0):
        
        old_net = self.net

        self.net = AdjLeNet(num_kernels_layer1= self.num_kernels_layer1 + added_kernels_layer1,
                           num_kernels_layer2= self.num_kernels_layer2 + added_kernels_layer2,
                           num_kernels_layer3= self.num_kernels_layer3 + added_kernels_layer3)

        if added_kernels_layer1 != 0:
            self.net.conv1.weight.data[0:self.num_kernels_layer1, :, :] = old_net.conv1.weight.data
            self.num_kernels_layer1 = self.num_kernels_layer1 + added_kernels_layer1

        if added_kernels_layer2 != 0:
            self.net.conv2.weight.data[0:self.num_kernels_layer2, :, :] = old_net.conv2.weight.data
            self.num_kernels_layer2 = self.num_kernels_layer2 + added_kernels_layer2
        
        if added_kernels_layer3 != 0:
            self.net.conv3.weight.data[0:self.num_kernels_layer3, :, :] = old_net.conv3.weight.data
            self.num_kernels_layer3 = self.num_kernels_layer3 + added_kernels_layer3

        print("\n\nExpanded Model Dimensions:")
        summary(self.net, input_size=(3, 32, 32), device= "cpu")



