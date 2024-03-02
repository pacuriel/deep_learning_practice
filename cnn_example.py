#imports
import torch #entire Pytorch library
import torch.nn as nn #NN modules 
import torch.optim as optim #optimization algos (SGD, ADAM, etc.)
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

#create simple CNN
class CNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(CNN, self).__init__()
        #Note: num_output_feats = floor((num_in_channels + 2*padding_size - kernel_size) / stride_size) + 1
        #first conv layer (same convolution)
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.fc1 = nn.Linear(16*7*7, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)

        return x

#sanity check
# model = CNN()
# x = torch.randn(64, 1, 28, 28)
# print(model(x).shape)
# exit()

#set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#hyperparameters
in_channels = 1
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 5

#load data
train_dataset = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True) #will download if we don't already have it
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = datasets.MNIST(root='dataset/', train=False, transform=transforms.ToTensor(), download=True) #will download if we don't already have it
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

#initialize network
model = CNN(in_channels=in_channels, num_classes=num_classes).to(device)

#loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

#train network
for epoch in range(num_epochs): #one epoch = NN has seen all images
    for batch_idx, (data, targets) in enumerate(train_loader): #data = image, target = label
        if batch_idx % 10 == 0:  
            print(f"Training on batch {batch_idx}")
     
        #move data/targets to device
        data = data.to(device=device)
        targets = targets.to(device=device)

        # data = data.reshape(data.shape[0], -1) #reshape data to single column
        # print(data.shape) #sanity check

        #forward pass
        scores = model(data)
        loss = criterion(scores, targets)

        #backward pass (backprop)
        optimizer.zero_grad() #set gradients to zero initially
        loss.backward() #update weights depending on gradients computed above

        #gradient descent (or adam step)
        optimizer.step()
        
        
#check accuracy on training and test data
def check_accuracy(loader, model):
    if loader.dataset.train: 
        print("Checking accuracy on training data")
    else:   
        print("Checking accuracy on test data")

    num_correct = 0 
    num_samples = 0 
    model.eval() #set model to evaluation mode

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            # x = x.reshape(x.shape[0], -1)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct) / float(num_samples)*100:.2f}')
        # acc = float(num_correct) / float(num_samples) #accuracy

    model.train() #set model back to training mode
    # return acc 

check_accuracy(train_loader, model)
check_accuracy(test_loader, model)