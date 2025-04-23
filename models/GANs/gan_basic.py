########################################################################################################################################
## Standard GAN Linear Model Boiler Plate !!!

## Author: Sayan Kr. Swar
## University of Rochester

#### Define Necessary Functions
#Contains 
# - GAN Model Functions
# - Train Loop Functions
# - Dataloading and Execution
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



class generator_basic(nn.Module):
    def __init__(self, inputdim, img_shape):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(inputdim,1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(1024,2048),
            nn.BatchNorm1d(2048),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.25),

            nn.Linear(2048,4096),
            nn.BatchNorm1d(4096),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.25),

            nn.Linear(4096, int(np.prod(img_shape))),
            nn.Tanh()

        )

    def forward(self, z):
        img = self.layers(z)
        img = img.view(img.size(0), *img_shape)
        return img


class discriminator_basic(nn.Module):
    def __init__(self,img_shape):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 4096),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.25),
            nn.Linear(4096, 2048),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.25),
            nn.Linear(2048, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.layers(img_flat)
        return validity


def train_aae(generator, discriminator, dataloader, device, latent_dim, num_epochs, lrg, lrd, batch_size):

    optimizer_g = torch.optim.Adam(generator.parameters(), lr=lrg)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=lrd)
    criterion = nn.BCELoss()
    generator_loss = []
    discriminator_loss = []

    generator.train()
    discriminator.train()
    
    for epoch in range(num_epochs):
        gloss=0.0
        dloss=0.0    
    
        for i, (images, _) in enumerate(dataloader):
            real_imgs = images.to(device)

            real_labels = torch.ones((images.size(0), 1)).to(device)
            fake_labels = torch.zeros((images.size(0), 1)).to(device)

            # Train discriminator
            optimizer_d.zero_grad()
            real_outputs = discriminator(real_imgs)
            real_loss = criterion(real_outputs, real_labels)

            fake_images = generator(torch.randn(images.size(0), latent_dim).to(device))
            fake_outputs = discriminator(fake_images.detach())
            fake_loss = criterion(fake_outputs, fake_labels)
            
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_d.step()


            # Train generator
            optimizer_g.zero_grad()
            fake_outputs = discriminator(fake_images)
            g_loss = criterion(fake_outputs, real_labels)
            g_loss.backward()
            optimizer_g.step()


            gloss+=g_loss.item()
            dloss+=d_loss.item()


        generator_loss.append(gloss/(i+1))
        discriminator_loss.append(dloss/(i+1))
        print(f"Epoch [{epoch + 1}/{num_epochs}]  D Loss: {(dloss/(i+1)):.4f}  G Loss: {(gloss/(i+1)):.4f}")
        
    return generator_loss, discriminator_loss

def imshow(img):
    npimg = torchvision.utils.make_grid(img).numpy()
    npimg = np.transpose(npimg, (1, 2, 0))
    plt.imshow(npimg)


if __name__ == "__main__":
    batch_size = 64
    num_epochs = 150
    latent_dim = 200
    lr_rate = 1e-4

    transform = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: F.interpolate(x.unsqueeze(0), size=(64, 64),  mode='bilinear', align_corners=False).squeeze(0)),
        transforms.Normalize((0.5,), (0.5,))
    ])

    traindata = datasets.ImageFolder(root='/data/chest_xray/train/', transform=transform)
    trainloader = DataLoader(traindata, batch_size=batch_size, shuffle=True, num_workers=24)

    dataiter = iter(trainloader)
    images, _ = next(dataiter)
    tmp_img, _ = next(dataiter)
    img_shape = tmp_img[1].numpy().shape

    generator = generator_basic(latent_dim,img_shape).to(device)
    discriminator = discriminator_basic(img_shape).to(device)

    generator_loss, discriminator_loss = train_aae(generator, discriminator, trainloader, device, latent_dim, num_epochs, lrg=lr_rate, lrd=lr_rate, batch_size=batch_size)
    

    generator.eval()
    fixed_noise = torch.randn(16, latent_dim, device=device)
    with torch.no_grad():
        fake = generator(fixed_noise).detach().cpu()
    grid = vutils.make_grid(fake, nrow=4, normalize=True)
    plt.figure(figsize=(8, 8))
    plt.imshow(grid.permute(1, 2,  0))
    plt.axis("off")
    plt.title("Generated Images from Random Noise")
