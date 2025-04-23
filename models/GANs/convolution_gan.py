########################################################################################################################################
## Covolutional GAN Model Boiler Plate !!!

## Author: Sayan Kr. Swar
## University of Rochester

#### Define Necessary Functions
#Contains 
# - Convolution Generator and Discriminator Model Functions
# - Train Loop Functions
########################################################################################################################################

import torch
import torchvision
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms,datasets
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import torchvision.utils as vutils



class Generator(nn.Module):
    def __init__(self, latent_dim, img_channels=1, features_g=64):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, features_g * 8, 4, 1, 0),  # 1x1 -> 4x4
            nn.BatchNorm2d(features_g * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(features_g * 8, features_g * 4, 4, 2, 1),  # 4x4 -> 8x8
            nn.BatchNorm2d(features_g * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(features_g * 4, features_g * 2, 4, 2, 1),  # 8x8 -> 16x16
            nn.BatchNorm2d(features_g * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(features_g * 2, features_g, 4, 2, 1),  # 16x16 -> 32x32
            nn.BatchNorm2d(features_g),
            nn.ReLU(True),

            nn.ConvTranspose2d(features_g, img_channels, 4, 4, 0),  # 32x32 -> 128x128
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)

# Discriminator
class Discriminator(nn.Module):
    def __init__(self, img_channels=1, features_d=64):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(img_channels, features_d, 4, 4, 0),  # 128x128 -> 32x32
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(features_d, features_d * 2, 4, 2, 1),  # 32x32 -> 16x16
            nn.BatchNorm2d(features_d * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(features_d * 2, features_d * 4, 4, 2, 1),  # 16x16 -> 8x8
            nn.BatchNorm2d(features_d * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(features_d * 4, features_d * 8, 4, 2, 1),  # 8x8 -> 4x4
            nn.BatchNorm2d(features_d * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(features_d * 8, 1, 4, 1, 0),  # 4x4 -> 1x1
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

def train_gan(G, D, dataloader, optimizer_G, optimizer_D, latent_dim, device, num_epochs=50):
    generator_loss = []
    discriminator_loss = []
    criterion = nn.BCELoss()

    for epoch in range(num_epochs):
        for real_imgs, _ in dataloader:
            real_imgs = real_imgs.to(device)
            real_imgs = real_imgs.mean(dim=1).unsqueeze(1)
            batch_size = real_imgs.size(0)

            # Train Discriminator
            noise = torch.randn(batch_size, latent_dim, 1, 1, device=device)
            fake_imgs = G(noise)

            real_labels = torch.ones(batch_size, device=device)
            fake_labels = torch.zeros(batch_size, device=device)

            optimizer_D.zero_grad()
            loss_real = criterion(D(real_imgs).view(-1), real_labels)
            loss_fake = criterion(D(fake_imgs.detach()).view(-1), fake_labels)
            d_loss = (loss_real + loss_fake) / 2
            d_loss.backward()
            optimizer_D.step()

            # Train Generator
            optimizer_G.zero_grad()
            output = D(fake_imgs).view(-1)
            g_loss = criterion(output, real_labels)
            g_loss.backward()
            optimizer_G.step()

        generator_loss.append(g_loss.item())    
        discriminator_loss.append(d_loss.item())
        print(f"Epoch [{epoch+1}/{num_epochs}]  D Loss: {d_loss.item():.4f}  G Loss: {g_loss.item():.4f}")

    print("Training Complete!")
    return generator_loss,discriminator_loss

def imshow(img):
    npimg = torchvision.utils.make_grid(img).numpy()
    npimg = np.transpose(npimg, (1, 2, 0))
    plt.imshow(npimg)

