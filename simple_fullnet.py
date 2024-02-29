#imports
import torch #entire Pytorch library
import torch.nn as nn #NN modules 
import torch.optim as optim #optimization algos (SGD, etc.)
import torch.nn.functional as F #activation functions, etc.
from torch.utils.data import DataLoader #helps with dataset management
import torchvision.datasets as datasets #include built-in datasets (MNIST, etc.)
import torchvision.transforms as transforms #used for data transormations

#create fully connected network
class NN(nn.Module):
    def __init__(self, input_size, num_classes): #(input_size = 28x28 = 784, num_classe=10)
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

#testing model output
# model = NN(784, 10)
# x = torch.randn(64, 784) #64 = num of examples (ie minibatch size)
# print(model(x).shape)
     
#set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#hyperparameters
input_size = 784 #mnist dims
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 1

#load data
train_dataset = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True) #will download if we don't already have it
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = datasets.MNIST(root='dataset/', train=False, transform=transforms.ToTensor(), download=True) #will download if we don't already have it
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

###left off ~9:30

#initialize network

#loss and optimizer

#train network

#check accuracy on training and test data