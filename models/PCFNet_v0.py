########################################################################################################################################
## PCFNet Mdoel created srom scratch that can utilize custom made filters on the initial convoltuiton layer and that improves specific
## imaging problems as per need. This model is the skeleton architecture and not yet fully optimized. Some updates are still required to 
## improve the speed for example "define_gabor_optim_filter" must be integrated into the "define_custom_filter" function.  
## "define_custom_filter" step needs another update for inclusion of additional filters in the first convoltuon layer step. 
# I will get back to it when I have some time.

## Author: Sayan Kr. Swar
## University of Rochester
## cite: https://www.sciencedirect.com/science/article/pii/S0925231219316789?via%3Dihub#sec0017

#### Define Necessary Functions
# - In this section defining the model architecture as provided in the question in the model ConvTestNet()
# - Function to call training for the model (without dropout): *ConvTestNet*
# - Function to call training for the model (with dropout): *ConvTestNet_Dropout*
# - Function to Run CNN with Predefined Filter Sets: *ConvTest_Predefined*
# - Function to test the result of the model training with both training and testing dataset
# - Function to load the MNIST dataset using dataloader.
# - Setting Seeds for reproducible outputs
########################################################################################################################################
import torch
import torchvision
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import datetime
import time
import os
import random
import cv2



class ConvTestNet(nn.Module):
    def __init__(self):
       super().__init__()
       self.conv1 = nn.Conv2d(1,32,kernel_size=3,padding=1)          #First Convolution layer
       self.conv2 = nn.Conv2d(32,64,kernel_size=3,padding=1)         #Second Convolution layer
       self.fc1 = nn.Linear(3136,128)
       self.fc2 = nn.Linear(128,10)

    def forward(self,x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out,kernel_size=2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out,kernel_size=2)
        out = torch.flatten(out,start_dim=1)
        out = F.relu(self.fc1(out))
        #out = F.softmax(self.fc2(out), dim=1)                       #cite: https://tinyurl.com/v2ft37wp
        out = self.fc2(out)
        return out

class ConvTestNet_Dropout(nn.Module):
    def __init__(self):
       super().__init__()
       self.conv1 = nn.Conv2d(1,32,kernel_size=3,padding=1)          #First Convolution layer
       self.conv2 = nn.Conv2d(32,64,kernel_size=3,padding=1)         #Second Convolution layer
       self.fc1 = nn.Linear(3136,128)
       self.dropout1 = nn.Dropout(0.25)
       self.fc2 = nn.Linear(128,10)

    def forward(self,x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out,kernel_size=2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out,kernel_size=2)
        out = torch.flatten(out,start_dim=1)
        out = F.relu(self.fc1(out))
        out = self.dropout1(out)
        #out = F.softmax(self.fc2(out), dim=1)                       #cite: https://tinyurl.com/v2ft37wp
        out = self.fc2(out)
        return out


class ConvTest_PCFNet(nn.Module):
    def __init__(self,ksize,num_filters_gab,num_filters_imap):
       super().__init__()
       self.theta = nn.Parameter(torch.tensor(np.random.uniform(0, 180, num_filters_gab), dtype=torch.float32),requires_grad=True)
       self.sigma = nn.Parameter(torch.tensor(np.random.uniform(0, 2*180, num_filters_gab), dtype=torch.float32),requires_grad=True)
       self.gamma = nn.Parameter(torch.tensor(np.random.uniform(0, 1, num_filters_gab), dtype=torch.float32),requires_grad=True)
       self.lambd = nn.Parameter(torch.tensor(np.random.uniform(2, 10, num_filters_gab), dtype=torch.float32),requires_grad=True)
       self.psi = nn.Parameter(torch.tensor(np.random.uniform(0, 0, num_filters_gab), dtype=torch.float32),requires_grad=True)
       self.ksize = ksize
       self.num_filters_gab = num_filters_gab
       self.num_filters_imap = num_filters_imap
       
       #self.conv1 = nn.Conv2d(1,num_filters_gab+num_filters_imap,
       #                        kernel_size=ksize,padding=1,stride=1)          #First Predefine Convolution layer

       self.conv2 = nn.Conv2d(32,64,kernel_size=3,padding=1)                 #Second Convolution layer
       self.fc1 = nn.Linear(3136,128)
       self.dropout1 = nn.Dropout(0.25)
       self.fc2 = nn.Linear(128,10)
       print(self.theta,self.sigma,self.gamma,self.lambd,self.psi)
    

    def forward(self,x):
        conv1_layer = self.define_custom_filter(self.ksize, self.num_filters_gab,self.num_filters_imap,
                                                      self.theta, self.sigma, self.gamma,self.lambd,self.psi)

        out = F.relu(F.conv2d(x,conv1_layer,bias=None,stride=1,padding=1)) #First Predefined Convolution Layer
        
        out = F.max_pool2d(out,kernel_size=2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out,kernel_size=2)
        out = torch.flatten(out,start_dim=1)
        out = F.relu(self.fc1(out))
        out = self.dropout1(out)
        #out = F.softmax(self.fc2(out), dim=1)                               #cite: https://tinyurl.com/v2ft37wp
        out = self.fc2(out)
        return out
    
    def define_custom_filter(self, kernel_size, num_filters, num_filters_imap, theta, sigma, gamma, lambd, psi):
        if sigma.shape[0]!=num_filters or theta.shape[0]!=num_filters or \
            lambd.shape[0]!=num_filters or gamma.shape[0]!=num_filters or \
            psi.shape[0]!=num_filters:
            raise Exception('Parameter Size and Num of Filters Do Not Match!')
        
        kern = torch.zeros([num_filters,kernel_size,kernel_size],dtype=torch.float32)
        for fn in range(0,num_filters):
            for x in range(0,kernel_size):
                for y in range(0,kernel_size):
                    x_theta = (x-1)*torch.cos(theta[fn])+(y-1)*torch.sin(theta[fn])
                    y_theta = -1*(x-1)*torch.sin(theta[fn])+(y-1)*torch.cos(theta[fn])
                    sigma_x = sigma[fn]
                    sigma_y = sigma[fn]/gamma[fn]
                    kern[fn,x,y] = torch.exp(-0.5 * (x_theta**2 / sigma_x**2 + y_theta**2 / sigma_y**2)) * torch.cos( (2 * torch.pi * x_theta) / lambd[fn] + psi[fn])

        imaps = np.array([[ 0, 0, 0,],[ 0, 1, 0,],[ 0, 0, 0,]])
        imaps = np.tile(imaps,(num_filters_imap,1,1))
        imaps = torch.tensor(imaps, dtype=torch.float32)
            
        first_conv_layer = torch.vstack((kern, imaps))
        first_conv_layer = first_conv_layer.unsqueeze(1)
        return first_conv_layer.cuda()
        

def training_loop(num_epochs,model,dataloader_train,dataloader_test,optimizer,criterion,device):
        loss_track=[]
        accu_track=[]
        val_accu_track=[]
        for epoch in range(1,num_epochs+1):
            loss_train=0.0
            total=0
            correct=0.0
            model.train()
            for data, targets in dataloader_train:
                data = data.to(device)
                targets = targets.to(device)
                optimizer.zero_grad()
                outputs = model(data)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                loss_train += loss.item()
                loss_track.append(loss.item())

                ##Train Accuracy Check
                _, op_label = torch.max(outputs,dim=1)
                total+=outputs.shape[0]
                correct+= int((op_label==targets).sum())
            accu_track.append(correct/total)

            ##Test Accuracy Check
            model.eval()
            totalv=0
            correctv=0
            with torch.no_grad():
                for img, label in dataloader_test:
                    img=img.to(device)
                    label=label.to(device)
                    val_output = model(img)
                    _, idx = torch.max(val_output, dim=1)
                    totalv+= label.shape[0]
                    correctv+= int((idx==label).sum())
            val_accu_track.append(correctv/totalv)
            
            if epoch==1 or epoch%10==0:
                print(f'{datetime.datetime.now()} Epoch {epoch}; Avg Loss in this Epoch: {(loss_train/len(dataloader_train))}; Train Accuracy: {accu_track[-1]}; Test Accuracy: {val_accu_track[-1]}')
        return loss_track,accu_track,val_accu_track



def evaluate_model(model,dataloader_train, dataloader_test, device):
    model.eval()
    with torch.no_grad():   
        for name, loader in [('train set',dataloader_train),('test set',dataloader_test)]:
            total=0
            correct=0
            for img, label in loader:
                img=img.to(device)
                label=label.to(device)
                val_output = model(img)
                _, idx = torch.max(val_output, dim=1)
                total+= label.shape[0]
                correct+= int((idx==label).sum())
            print("Accuracy {}: {:.2f}".format(name, correct/total))


def mnist_data_load(path,transform,batch_size=128):
    train_dataset = torchvision.datasets.MNIST(root=f'{path}/data',train=True, transform=transform, download=True)
    test_dataset = torchvision.datasets.MNIST(root=f'{path}/data',train=False,transform=transform, download=True)

    dataloader_train = DataLoader(train_dataset,batch_size=batch_size, shuffle=True, num_workers=16)
    dataloader_test = DataLoader(test_dataset,batch_size=batch_size, shuffle=False, num_workers=16)
    return dataloader_train,dataloader_test

def gabor_kernel_old(ksize=16):
    theta = np.random.uniform(0, 2)
    sigma = np.random.uniform(0, 2*np.pi)
    gamma = np.random.uniform(0, 1)
    lambd = np.random.uniform(2, 10)
    psi = np.random.uniform(-np.pi, np.pi)
    return cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_32F)

def genrate_predefined_filters_old(ksize, num_filters_gab,num_filters_imap):
    kernels = []
    for _ in range(0,num_filters_gab):
        kernels.append(gabor_kernel_old(ksize))
    kernels = np.array(kernels)
    
    imaps = np.array([[ 0, 0, 0,],[ 0, 1, 0,],[ 0, 0, 0,]])
    imaps = np.tile(imaps,(num_filters_imap,1,1))
        
    first_conv_layer = torch.tensor(np.vstack((kernels, imaps)), dtype=torch.float32)
    first_conv_layer = first_conv_layer.unsqueeze(1)
    return first_conv_layer

def genrate_predefined_filters_opencv(ksize, num_filters_gab, num_filters_imap, sigma, theta, lambd, gamma, psi):
    kernels = []
    for _ in range(0,num_filters_gab):
        kernels.append(cv2.getGaborKernel((ksize, ksize), sigma.item(), theta.item(), 
                                            lambd.item(), gamma.item(), psi.item(), 
                                            ktype=cv2.CV_32F))
    kernels = np.array(kernels)
    
    imaps = np.array([[ 0, 0, 0,],[ 0, 1, 0,],[ 0, 0, 0,]])
    imaps = np.tile(imaps,(num_filters_imap,1,1))
        
    first_conv_layer = torch.tensor(np.vstack((kernels, imaps)), dtype=torch.float32)
    first_conv_layer = first_conv_layer.unsqueeze(1)
    return first_conv_layer.cuda()

def define_gabor_optim_filter(kernel_size, num_filters, theta, sigma, gamma, lambd, psi):
    #### Optimized Gabor Filter following PCFNet, Using Vectorized Approach
    if any(param.shape[0] != num_filters for param in [sigma, theta, lambd, gamma, psi]):
        raise ValueError("Parameter size and number of filters do not match!")

    coords = torch.arange(-(kernel_size // 2), kernel_size // 2 + 1, dtype=torch.float32)
    x, y = torch.meshgrid(coords, coords, indexing='ij')  # Create 2D grid

    x, y = x.unsqueeze(0), y.unsqueeze(0)

    x_theta = x * torch.cos(theta[:, None, None]) + y * torch.sin(theta[:, None, None])
    y_theta = -x * torch.sin(theta[:, None, None]) + y * torch.cos(theta[:, None, None])

    sigma_x = sigma[:, None, None]
    sigma_y = sigma_x / gamma[:, None, None]

    gaussian_envelope = torch.exp(-0.5 * (x_theta**2 / sigma_x**2 + y_theta**2 / sigma_y**2))
    sinusoidal_carrier = torch.cos((2 * torch.pi * x_theta) / lambd[:, None, None] + psi[:, None, None])

    kern = gaussian_envelope * sinusoidal_carrier

    return kern  # Shape: (num_filters, kernel_size, kernel_size)

def set_seed(seed: int = 0) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")
set_seed()



#################################################################################################################################
## Running the Model
#################################################################################################################################

# transform = transforms.Compose([
#             transforms.ToTensor()
#     ])

# data_path = './PRJ_DL_PRACTICE/ECE_484_Coursework/HW2'
# dataloader_train,dataloader_test = mnist_data_load(data_path,transform, 128)
# data_batch = next(iter(dataloader_train))
# print('A Single Image Size:', data_batch[0].shape)

# device  = torch.device("cuda:0" if torch.cuda.is_available ()
#                        else "mps" if torch.backends.mps.is_available()
#                        else "cpu")

# transform = transforms.Compose([
#             transforms.ToTensor()])

# data_path = './PRJ_DL_PRACTICE/ECE_484_Coursework/HW2'
# dataloader_train,dataloader_test = mnist_data_load(data_path,transform, 128)

# kernel_size=3
# num_gabor_filters=24
# num_imap=8
# model_predef = ConvTest_Predefined(kernel_size,num_gabor_filters,num_imap).to(device)
# criterion = nn.CrossEntropyLoss().to(device)
# optimizer = optim.Adam(model_predef.parameters(), lr=0.001)

# num_epochs=10

# loop_time=time.time()
# loss_track_predef,accu_track_predef,accu_test_track_predef = training_loop(num_epochs,model_predef,
#                                                                            dataloader_train,dataloader_test,
#                                                                            optimizer,criterion,device)

# print(f'Traning Time (seconds) for {num_epochs} Epochs: {time.time()-loop_time}')

# evaluate_model(model_predef,dataloader_train,dataloader_test,device)

# plt.figure(figsize=(14,4))
# plt.subplot(1,3,1)
# plt.plot(loss_track_predef, label='train loss')
# plt.title('Plot Model Loss Over Iterations')
# plt.ylabel('Loss')
# plt.xlabel('Iterations')

# plt.subplot(1,3,2)
# plt.plot(accu_track_predef, label='train accuracy')
# plt.title('Plot Model Train Accuracy Over Epochs')
# plt.ylabel('Train Accuracy')
# plt.xlabel('Epoch')

# plt.subplot(1,3,3)
# plt.plot(accu_test_track_predef, label='test accuracy')
# plt.title('Plot Model Test Accuracy Over Epochs')
# plt.ylabel('Test Accuracy')
# plt.xlabel('Epoch')
# plt.show()

# print('---------------------CNN Base Model-----------------------------------')
# for name, param in model_main.named_parameters():
#     print(f'{name}: {param.requires_grad}')

# numel_list_predef = [p.numel() for p in model_main.parameters() if p.requires_grad]
# print('Total Num of Parameters: ', sum(numel_list_predef))
# print('Parameters in Each Layer: ', numel_list_predef)
# print('')
# print('---------------------CNN Predefined Filters-----------------------------------')
# for name, param in model_predef.named_parameters():
#     print(f'{name}: {param.requires_grad}')

# numel_list_predef = [p.numel() for p in model_predef.parameters() if p.requires_grad]
# print('Total Num of Parameters: ', sum(numel_list_predef))
# print('Parameters in Each Layer: ', numel_list_predef)