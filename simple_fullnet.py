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

#initialize network
model = NN(input_size=input_size, num_classes=num_classes).to(device)

#loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

#train network
for epoch in range(num_epochs): #one epoch = NN has seen all images
    for batch_idx, (data, targets) in enumerate(train_loader): #data = image, target = label
        #move data/targets to device
        data = data.to(device=device)
        targets = targets.to(device=device)

        data = data.reshape(data.shape[0], -1) #reshape data to single column
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
    if loader.dataset.train: k

    num_correct = 0
    num_samples = 0
    model.eval() #set model to evaluation mode

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            x = x.reshape(x.shape[0], -1)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct) / float(num_samples)*100:.2f}')

    model.train() #set model back to training mode
    return acc 

check_accuracy(train_loader, model)
check_accuracy(test_loader, model)