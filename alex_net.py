###Implementing a version of the AlexNet (2012) model applied to the Imagenette dataset (subset of ImageNet)

#importing packages
import torch #entire Pytorch library
import torch.nn as nn #NN modules 
import torch.optim as optim #optimization algos (SGD, ADAM, etc.)
import torch.nn.functional as F #activation functions, etc.
from torch.utils.data import DataLoader #helps with dataset management
import torchvision.datasets as datasets #include built-in datasets (MNIST, etc.)
import torchvision.transforms as transforms #used for data transormations

#defining AlexNet model
class AlexNet(nn.Module):
    #in_channels=3 bc RGB images
    def __init__(self, in_channels=3, num_classes=1000):
        super(AlexNet, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=(3,3), stride=2) #pooling layer
        self.dropout = nn.Dropout(p=0.5) #dropout layer
        #first conv layer
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=96, kernel_size=(11,11), stride=4, padding=1)
        #second conv layer
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=(5,5), stride=1, padding=2)
        #third conv layer
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=(3,3), stride=1, padding=1)
        #fourth conv layer
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=(3,3), stride=1, padding=1)
        #fifth conv layer
        self.conv5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=(3,3), stride=1, padding=1)
        
        #FC layers
        self.fc1 = nn.Linear(in_features=256*5*5, out_features=4096) #first FC layer
        self.fc2 = nn.Linear(in_features=4096, out_features=4096) #second FC layer
        self.fc3 = nn.Linear(in_features=4096, out_features=num_classes) #third FC layer

    #forward pass of LeNet
    def forward(self, x):
        #conv/pool layers
        x = F.relu(self.conv1(x)) #applying activation function to output from first conv layer 
        x = self.pool(x) #pooling layer
        x = F.relu(self.conv2(x)) #act fcn applied to output from second conv layer
        x = self.pool(x) #pooling layer
        x = F.relu(self.conv3(x)) #act fcn to output of third conv layer
        x = F.relu(self.conv4(x)) #act fcn to output of fourth conv layer
        x = F.relu(self.conv5(x)) #act fcn to output of fifth conv layer
        x = self.pool(x) #pooling layer

        #FC layers
        x = torch.flatten(x, 1) #flattening data to input into FC layers 
        x = F.relu(self.fc1(x)) #act fcn on first FC layer
        x = self.dropout(x) #applying dropout
        x = F.relu(self.fc2(x)) #second FC layer
        x = self.dropout(x) #applying dropout
        x = self.fc3(x) #third FC/output layer
        return x

#sanity check
# model = AlexNet()
# x = torch.randn(1, 3, 224, 224)
# print(model(x).shape)
# x = model(x)
# print(x.shape)
# exit()

#set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#hyperparameters
in_channels = 3 #binary images (3 if RGB)
num_classes = 10
learning_rate = 0.001 #LR used for optimizer
batch_size = 64 #size of each batch of data to train on
num_epochs = 5 #number epochs to train for

#load data
train_dataset = datasets.Imagenette(root='datasets/', split="train", transform=transforms.ToTensor(), download=True) #will download if we don't already have it
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = datasets.Imagenette(root='datasets/', split="val", transform=transforms.ToTensor(), download=True) #will download if we don't already have it
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

#initialize network
model = AlexNet(in_channels=in_channels, num_classes=num_classes).to(device)

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