###Implementing a modified version of the classic LeNet-5 (1998) model 
###and applying it to FashionMNIST dataset.
#importing packages
import torch #entire Pytorch library
import torch.nn as nn #NN modules 
import torch.optim as optim #optimization algos (SGD, ADAM, etc.)
import torch.nn.functional as F #activation functions, etc.
from torch.utils.data import DataLoader #helps with dataset management
import torchvision.datasets as datasets #include built-in datasets (MNIST, etc.)
import torchvision.transforms as transforms #used for data transormations

#defining LeNet model
class LeNet(nn.Module):
    #in_channels=1 bc grayscale/binary images
    def __init__(self, in_channels=1, num_classes=10):
        super(LeNet, self).__init__()
        #first conv layer
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=6, kernel_size=(5,5), stride=1, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=2) #pooling layer
        #second conv layer
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(5,5), stride=1)
        #FC layers
        self.fc1 = nn.Linear(in_features=16*5*5, out_features=120) #first FC layer
        self.fc2 = nn.Linear(in_features=120, out_features=84) #second FC layer
        self.fc3 = nn.Linear(in_features=84, out_features=num_classes) #third FC layer

    #forward pass of LeNet
    def forward(self, x):
        x = F.relu(self.conv1(x)) #applying activation function to output from first conv layer 
        x = self.pool(x) #pooling layer
        x = F.relu(self.conv2(x)) #act fcn applied to output from second conv layer
        x = self.pool(x) #pooling layer
        x = torch.flatten(x, 1) #flattening data to input into FC layers 
        x = F.relu(self.fc1(x)) #first FC layer
        x = F.relu(self.fc2(x)) #second FC layer
        x = self.fc3(x) #third FC layer
        return x 

#sanity check
# model = LeNet()
# x = torch.randn(1, 1, 28, 28)
# print(x.shape)
# print(x[0].shape)
# print(x[0])
# x = torch.flatten(x, 1)
# print(x.shape)
# print(model(x).shape)
# print(x.reshape(x, -1).shape)
# exit()

#set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#hyperparameters
in_channels = 1 #binary images (3 if RGB)
num_classes = 10
learning_rate = 0.001 #LR used for optimizer
batch_size = 64 #size of each batch of data to train on
num_epochs = 5 #number epochs to train for

#load data
train_dataset = datasets.FashionMNIST(root='datasets/', train=True, transform=transforms.ToTensor(), download=True) #will download if we don't already have it
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = datasets.FashionMNIST(root='datasets/', train=False, transform=transforms.ToTensor(), download=True) #will download if we don't already have it
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

#initialize network
model = LeNet(in_channels=in_channels, num_classes=num_classes).to(device)

#loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

#train network
for epoch in range(num_epochs): #one epoch = NN has seen ALL images
    print(f'*** Began training on epoch {epoch + 1}  ***')
    #looping over each batch of data
    for batch_idx, (data, targets) in enumerate(train_loader): #data = image, target = label
        if batch_idx % 50 == 0:
            print(f"Training on batch {batch_idx}") #sanity check
     
        #move data/targets to device
        data = data.to(device=device)
        targets = targets.to(device=device)

       
        scores = model(data) #forward pass
        loss = criterion(scores, targets) #calculating loss on one batch of data (64 imgs)

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

#printing accuracy on train and test sets
check_accuracy(train_loader, model)
check_accuracy(test_loader, model)