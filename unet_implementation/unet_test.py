
#imports
import torch #entire Pytorch library
import torch.nn as nn #NN modules 
import torch.optim as optim #optimization algos (SGD, ADAM, etc.)
import torch.nn.functional as TF #activation functions, etc.
from torch.utils.data import DataLoader #helps with dataset management
import torchvision.datasets as datasets #include built-in datasets (MNIST, etc.)
import torchvision.transforms as transforms #used for data transormations

#class to run double convs used in UNET
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False), #bias = false bc using batch norm in next line
            nn.BatchNorm2d(out_channels), #note: original u-net paper does not use batch norm (wasn't invented yet)
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x) #applies above sequential container to x

#creating U-Net model
class UNET(nn.Module):
    #Note: original paper has out_channels=2; 
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) #note: kernel_size = filter_size

        #down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature
        
        #up part of UNET
        #Note: will be using transpose convolutions for upsampling
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    in_channels=2*feature, out_channels=feature, kernel_size=2, stride=2)
            )#append
            self.ups.append(DoubleConv(in_channels=2*feature, out_channels=feature))

        #bottleneck layer (bottom of architecture diagram)
        self.bottleneck = DoubleConv(in_channels=features[-1], out_channels=2*features[-1])

        #final conv layer
        self.final_conv = nn.Conv2d(in_channels=features[0], out_channels=out_channels, kernel_size=1) 

    def forward(self, x): 
        skip_connections = [] #used to store skip connections
        
        for down in self.downs:
            x = down(x)
            skip_connections.append(x) #storing skip connection to use later
            x = self.pool(x)

        #bottleneck layer
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1] #reversing skip connection list

        #loop to do up part and double conv (step of 2)
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2] #storing skip connection

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:]) #resizing x to match shape of skip connection 

            concatenate_skip = torch.cat((skip_connection, x), dim=1) #concatentating skip connection
            x = self.ups[idx + 1](concatenate_skip) #running through double conv
        
        return self.final_conv(x)

#function to test UNET model
def test():
    x = torch.randn((3, 1, 160, 160)) #dummy variable to represent an RGB picture
    model = UNET(in_channels=1, out_channels=1)
    preds = model(x)
    print(preds.shape)
    print(x.shape)
    assert preds.shape == x.shape

if __name__ == "__main__":
    test()

###left off at ~22:00
###next steps:
    #download dataset (on Kaggle) and store in parent folder
    #build dataloader
    #test on data


# #set device
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#hyperparameters