#imports
import torch #entire Pytorch library
import torch.nn as nn #NN modules 
import torch.optim as optim #optimization algos (SGD, etc.)
import torch.nn.functional as F #activation functions, etc.
from torch.utils.data import DataLoader #helps with dataset management
import torchvision.datasets as datasets #include built-in datasets (MNIST, etc.)
import torchvision.transforms as transforms #used for data transormations



#testing model output
# model = NN(784, 10)
# x = torch.randn(64, 784) #64 = num of examples (ie minibatch size)
# print(model(x).shape)
     
#set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#hyperparameters
input_size = 28 #mnist dims
sequence_length = 28
num_layers = 2
hidden_size = 256 #num nodes in hidden layer
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 2

#create a RNN
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        #batch_size*time_seq*features
        self.fc = nn.Linear(hidden_size*sequence_length, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        #forward prop
        out, _ = self.rnn(x, h0)
        out = out.reshape(out.shape[0], -1)
        out = self.fc(out)
        return out
    
#create a GRU (basically same code as RNN)
class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        #batch_size*time_seq*features
        self.fc = nn.Linear(hidden_size*sequence_length, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        #forward prop
        out, _ = self.gru(x, h0)
        out = out.reshape(out.shape[0], -1)
        out = self.fc(out)
        return out

#load data
train_dataset = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True) #will download if we don't already have it
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = datasets.MNIST(root='dataset/', train=False, transform=transforms.ToTensor(), download=True) #will download if we don't already have it
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

#initialize network
model = RNN(input_size, hidden_size, num_layers, num_classes).to(device)

#loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

#train network
for epoch in range(num_epochs): #one epoch = NN has seen all images
    for batch_idx, (data, targets) in enumerate(train_loader): #data = image, target = label
        #displaying sanity check
        if batch_idx % 50 == 0:  
            print(f"Training on batch {batch_idx}")
     
        #move data/targets to device
        data = data.to(device=device).squeeze(1)
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
            # print(x.shape)
            x = x.to(device=device).squeeze(1)
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