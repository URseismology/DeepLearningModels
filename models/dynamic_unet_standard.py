
########################################################################################################################################
## Dynamic Unet Advanced Class towards making the architecture more Dynamic and adjustable as per user need
## Author: Sayan Kr. Swar
## University of Rochester
## cite: https://arxiv.org/pdf/1505.04597
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

class unet_convlayers(nn.Module):
    def __init__(self, inchan:int, outchan:int, kernelsize=3, stride=1, loops=2):
        super().__init__()
        self.convlayers = nn.ModuleList([nn.Conv2d(inchan,outchan,kernelsize,stride) if i==0 else nn.Conv2d(outchan,outchan,kernelsize,stride) for i in range(0,loops)])

    def forward(self,x):
        for idx,clayer in enumerate(self.convlayers):
            x = F.relu(clayer(x))
        return x

class unet_downsample(nn.Module):
    def __init__(self,ipchan,chnlist=[64,128,256,512,1024],maxpoolsize=2,maxpoolstride=2):
        super().__init__()
        self.maxpool = nn.MaxPool2d(maxpoolsize,maxpoolstride)
        self.downsamplelayers = nn.ModuleList([unet_convlayers(ipchan,chnlist[idx]) if idx==0 else unet_convlayers(chnlist[idx-1],chnlist[idx]) for idx in range(0,len(chnlist))])
        
    def forward(self,x):
        contracting_path = []
        total_layers = len(self.downsamplelayers)
        for idx,downsample_layer in enumerate(self.downsamplelayers):
            x = downsample_layer(x)
            if idx < total_layers-1:
                contracting_path.append(x)
                x = self.maxpool(x)
        return x, contracting_path


class unet_upsample(nn.Module):
    def __init__(self,chnlist=[64,128,256,512,1024]):
        super().__init__()
        chnlist_rev = chnlist.copy()
        chnlist_rev.reverse()
        self.upsamplestep = nn.ModuleList([nn.ConvTranspose2d(chnlist_rev[idx],chnlist_rev[idx],kernel_size=2,stride=2) for idx in range(0,len(chnlist_rev)-1)])
        self.upsamplelayers = nn.ModuleList([unet_convlayers(chnlist_rev[idx]+chnlist_rev[idx+1],chnlist_rev[idx+1]) for idx in range(0,len(chnlist_rev)-1)])
        
    def forward(self,x,contracting_path):
        for idx, upsample_layer in enumerate(self.upsamplelayers):
            x = self.upsamplestep[idx](x)
            contracting_data = torchvision.transforms.functional.center_crop(contracting_path.pop(), [x.shape[2], x.shape[3]])
            residual_concated = torch.cat([x, contracting_data], dim=1)
            x = upsample_layer(residual_concated)

        return x

class unet_all(nn.ModuleList):
    def __init__(self, ipchan=1, chnlist=[64,128,256,512,1024], output_class=2):
        super().__init__()
        self.downsample_block = unet_downsample(ipchan,chnlist,maxpoolsize=2,maxpoolstride=2)
        self.upsample_block = unet_upsample(chnlist)
        self.output_layer = nn.Conv2d(chnlist[0],output_class,kernel_size=3,stride=1,padding=1)

    def forward(self,x):
        x, x_contracting = self.downsample_block(x)
        x = self.upsample_block(x, x_contracting)
        out = self.output_layer(x)

        return out

#### Testing the Sanity of the Model Architecture
# x = torch.randn(4, 1, 512, 512)#torch.randn(1, 3, 572, 572) #torch.randn(1, 1024, 28, 28)
# unet1 = unet_all(ipchan=1,chnlist=[32,64,128,256,512],output_class=2)

# uout1 = unet1(x)
# print(uout1.shape)