########################################################################################################################################
## Dynamic Segnet Advanced Class towards making the architecture more Dynamic and adjustable as per user need
## Author: Sayan Kr. Swar
## University of Rochester
## cite1: https://arxiv.org/pdf/1409.1556; 
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

class segnet_convlayers(nn.Module):
    def __init__(self, inchan:int, outchan:int, kernelsize=3, stride=1, padd=1, depth=2, direc='encode'):
        super().__init__()
        self.direc = direc
        self.convlayers_ec = nn.ModuleList([nn.Conv2d(inchan,outchan,kernelsize,stride,padd) if i==0 else nn.Conv2d(outchan,outchan,kernelsize,stride,padd) for i in range(0,depth)])
        self.convlayers_dc = nn.ModuleList([nn.Conv2d(inchan,outchan,kernelsize,stride,padd) if i==depth-1 else nn.Conv2d(inchan,inchan,kernelsize,stride,padd) for i in range(0,depth)])
        self.batchnorm_ec = nn.BatchNorm2d(outchan, momentum=0.5)
        self.batchnorm_dc = nn.BatchNorm2d(inchan, momentum=0.5)

    def forward(self,x):
        if self.direc == 'encode':
            for idx,clayer in enumerate(self.convlayers_ec):
                x = F.relu(self.batchnorm_ec(clayer(x)))
        elif self.direc == 'decode':
            cnvlyr_len = len(self.convlayers_dc)
            for idx,clayer in enumerate(self.convlayers_dc):
                x = clayer(x)
                if idx < cnvlyr_len-1:
                    x = self.batchnorm_dc(x)
                else:
                    x = self.batchnorm_ec(x)
        else:
            raise('AssertionError')
        return x


class segnet_encoder(nn.Module):
    def __init__(self,ipchan=3,chnlist=[64,128,256,512,512]):
        super().__init__()
        self.seg_encoder_layer = nn.ModuleList([segnet_convlayers(ipchan, chnlist[idx], depth=2, direc='encode') if idx==0 
                                                else segnet_convlayers(chnlist[idx-1], chnlist[idx], depth=2, direc='encode') if idx==1
                                                else  segnet_convlayers(chnlist[idx-1], chnlist[idx], depth=3, direc='encode') for idx in range(0,len(chnlist))])
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2, return_indices=True)

    def forward(self,x):
        mpl_idx_lst = []
        x_size_list = []
        for idx, encoder_layer in enumerate(self.seg_encoder_layer):
            x = encoder_layer(x)
            x, mp_idx = self.pool(x)
            x_size_list.append(x.shape)
            mpl_idx_lst.append(mp_idx)
        return x, mpl_idx_lst, x_size_list


class segnet_decoder(nn.Module):
    def __init__(self,chnlist=[64,128,256,512,512]):
        super().__init__()
        chnlist_rev = chnlist.copy()
        chnlist_rev.reverse()
        self.mpool_idx_mem = nn.MaxUnpool2d(2, stride=2)
        self.seg_decoder_layer = nn.ModuleList([segnet_convlayers(chnlist_rev[idx],chnlist_rev[idx+1], depth=3, direc='decode') if (idx>=0 and idx<3)
                                                else segnet_convlayers(chnlist_rev[idx],chnlist_rev[idx+1], depth=2, direc='decode') for idx in range(0,len(chnlist_rev)-1)])
                
    def forward(self, x, mpl_idx_lst, x_size_list):
        for idx, decoder_layer in enumerate(self.seg_decoder_layer):
            if len(x_size_list)>0:
                x = self.mpool_idx_mem(x,mpl_idx_lst.pop(), output_size=x_size_list.pop())
            else:
                x = self.mpool_idx_mem(x,mpl_idx_lst.pop())
            x = decoder_layer(x)
        return x


class segnet_all(nn.ModuleList):
    def __init__(self, ipchan=1, chnlist=[64,128,256,512,1024], output_chn=32):
        super().__init__()
        self.encoder_block = segnet_encoder(ipchan,chnlist)
        self.decoder_block = segnet_decoder(chnlist)
        #self.output_layer =  nn.Conv2d(chnlist[0],output_chn,kernel_size=3,stride=1,padding=1)

        self.output_layer =  segnet_convlayers(chnlist[0], output_chn, depth=3, direc='decode')

    def forward(self,x):
        x, mpl_idx_lst, x_size_list = self.encoder_block(x)
        x = self.decoder_block(x,mpl_idx_lst, x_size_list[0:-1])
        out = self.output_layer(x)
        out = F.softmax(out,dim=1)
        return out
    
#### Testing the Sanity of the Model Architecture
# x = torch.randn(1, 3, 572, 572) #torch.randn(1, 1024, 28, 28)
# segnet1 = segnet_all(ipchan=3,chnlist=[64,128,256,512,512],output_chn=32)

# segout1 = segnet1(x)
# print(segout1.shape)