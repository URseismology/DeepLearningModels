import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix
from skimage.metrics import structural_similarity as ssim


import os
import random

class vae_scratch(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.GELU(),

            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),

            nn.Conv2d(32, 64, 3, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Flatten()
        )

        self.z_mean = nn.Linear(1600,160)
        self.z_sd = nn.Linear(1600,160)

        self.decoder = nn.Sequential(
            nn.Linear(160,1600),
            nn.Unflatten(1, (64, 5, 5)),

            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(64,32,4,stride=1),
            nn.BatchNorm2d(32),
            nn.GELU(),

            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(32,16,3,stride=1,padding=1),
            nn.BatchNorm2d(16),
            nn.GELU(),

            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(16,1,3,stride=1,padding=1),
            nn.Tanh()
        ) 

    def encoding_from_reparam(self,x):
        x = self.encoder(x)
        z_mean, z_sd = self.z_mean(x), self.z_sd(x)
        x_latent = self.reparameterization(z_mean,z_sd)
        return x_latent, z_mean, z_sd

    def reparameterization(self,z_mu,z_sd):
        eps = torch.randn_like(z_mu).cuda()
        z = z_mu + eps * torch.exp(z_sd * 0.5)
        return z

    def forward(self,x_latent):
        x_latent, z_mean, z_sd = self.encoding_from_reparam(x_latent)
        x_decode = self.decoder(x_latent)
        return x_latent, z_mean, z_sd, x_decode



def training_loop(model,epochs,dataloader_train,criterion,optimizer,device):
    loss_track=[]

    for epoch in range(0,epochs):
        loss_item = 0
        for img,_ in train_loader:
            img = img.to(device)
            
            optimizer.zero_grad()

            out_latent, z_mean, z_sd, out_decode = model(img)
            kl_div = -0.5 * torch.sum(1 + z_sd - z_mean**2 - torch.exp(z_sd), axis=1)
            batchsize = kl_div.size(0)
            kl_div = kl_div.mean()
            
            pixelwise = criterion(out_decode,img,reduction='none')
            pixelwise = pixelwise.view(batchsize, -1).sum(axis=1) # sum over pixels
            pixelwise = pixelwise.mean()
            loss = 1*pixelwise + kl_div

            loss.backward()
            optimizer.step()

        if epoch%1==0 or epoch%2==0:
            print(f'Epoch={epoch} and Loss={loss.item():.4f}')
        loss_track.append(loss.item())

    return loss_track

def output_label(label):
    output_mapping = {
                 0: "T-shirt/Top",
                 1: "Trouser",
                 2: "Pullover",
                 3: "Dress",
                 4: "Coat", 
                 5: "Sandal", 
                 6: "Shirt",
                 7: "Sneaker",
                 8: "Bag",
                 9: "Ankle Boot"
                 }
    input = (label.item() if type(label) == torch.Tensor else label)
    return output_mapping[input]


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_set = torchvision.datasets.FashionMNIST("./data", download=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,))]))
    test_set = torchvision.datasets.FashionMNIST("./data", download=True, train=False, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,))]))
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=256, shuffle=True, num_workers=16)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=256, shuffle=False, num_workers=16)

    num_epochs = 50

    model = vae_scratch().to(device)
    criterion = F.mse_loss
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_track = training_loop(model,num_epochs,train_loader,criterion,optimizer,device)
