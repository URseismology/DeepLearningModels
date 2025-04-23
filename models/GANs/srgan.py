########################################################################################################################################
## SRGAN Model Boiler Plate !!!

## Author: Sayan Kr. Swar
## University of Rochester

#### Define Necessary Functions
#Contains 
# - SRGAN Model Functions
# - Train Loop Functions
# - Dataload function for High and Low Resolution Dataset curation
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


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        vgg19_model = vgg19(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(vgg19_model.features.children())[:18])

    def forward(self, img):
        return self.feature_extractor(img)


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_features, 0.8),
            nn.PReLU(),
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_features, 0.8),
        )

    def forward(self, x):
        return x + self.conv_block(x)


class GeneratorResNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_residual_blocks=16):
        super(GeneratorResNet, self).__init__()

        # First layer
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, 64, kernel_size=9, stride=1, padding=4), nn.PReLU())

        # Residual blocks
        res_blocks = []
        for _ in range(n_residual_blocks):
            res_blocks.append(ResidualBlock(64))
        self.res_blocks = nn.Sequential(*res_blocks)

        # Second conv layer post residual blocks
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(64, 0.8))

        # Upsampling layers
        upsampling = []
        for out_features in range(2):
            upsampling += [
                # nn.Upsample(scale_factor=2),
                nn.Conv2d(64, 256, 3, 1, 1),
                nn.BatchNorm2d(256),
                nn.PixelShuffle(upscale_factor=2),
                nn.PReLU(),
            ]
        self.upsampling = nn.Sequential(*upsampling)

        # Final output layer
        self.conv3 = nn.Sequential(nn.Conv2d(64, out_channels, kernel_size=9, stride=1, padding=4), nn.Tanh())

    def forward(self, x):
        out1 = self.conv1(x)
        out = self.res_blocks(out1)
        out2 = self.conv2(out)
        out = torch.add(out1, out2)
        out = self.upsampling(out)
        out = self.conv3(out)
        return out

class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()

        in_channels, in_height, in_width = input_shape

        def discriminator_block(in_filters, out_filters, first_block=False):
            layers = []
            layers.append(nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=1, padding=1))
            if not first_block:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.Conv2d(out_filters, out_filters, kernel_size=3, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        layers = []
        in_filters = in_channels
        for i, out_filters in enumerate([64, 128, 256, 512]):
            layers.extend(discriminator_block(in_filters, out_filters, first_block=(i == 0)))
            in_filters = out_filters

        self.model = nn.Sequential(
            *layers,
            nn.Flatten(),
            nn.Linear(512 * (in_height // 16) * (in_width // 16), 1)
        )

    def forward(self, img):
        return self.model(img)



# Training Function
def train_gan(G, D, dataloader, optimizer_G, optimizer_D, device, num_epochs=50):
    G.to(device)
    D.to(device)
    G.train()
    D.train()
    criterion_GAN = nn.MSELoss()
    criterion_content = nn.L1Loss()
    feature_extractor = FeatureExtractor().cuda()


    generator_loss = []
    discriminator_loss = []

    
    for epoch in range(num_epochs):
        for idx, (high_res,low_res) in enumerate(dataloader):
            high_res = high_res.to(deivce)
            low_res = low_res.to(deivce)

            valid = torch.ones((imgs.size(0), *D.output_shape), device=device, requires_grad=False)
            fake = torch.zeros((imgs.size(0), *D.output_shape), device=device, requires_grad=False)

            #  Train Generators
            optimizer_G.zero_grad()
            gen_hr = G(low_res) 
            loss_GAN = criterion_GAN(D(gen_hr), valid)
            gen_features = feature_extractor(gen_hr)
            real_features = feature_extractor(imgs)
            loss_content = criterion_content(gen_features, real_features.detach())
            loss_G = loss_content + 1e-3 * loss_GAN

            loss_G.backward()
            optimizer_G.step()

            #  Train Discriminator
            optimizer_D.zero_grad()
            loss_real = criterion_GAN(D(high_res), valid)
            loss_fake = criterion_GAN(D(gen_hr.detach()), fake)
            loss_D = (loss_real + loss_fake) / 2

            loss_D.backward()
            optimizer_D.step()

        generator_loss.append(loss_G.item())    
        discriminator_loss.append(loss_D.item())

        print(f"Epoch [{epoch + 1}/{num_epochs}]  D Loss: {loss_D.item():.4f}  G Loss: {loss_G.item():.4f}")

    print("Training Complete")
    return generator_loss,discriminator_loss


class read_covid_chest_xray(torch.utils.data.Dataset):
    def __init__(self, datapath='data/chest_xray/train'):
        print('By Default this function is going to read the Train Dataset unless provided otherwise')
        self.datapath = datapath
        data_class_paths = glob.glob(os.path.join(datapath,'*'))
        self.num_data_classes = len(data_class_paths)
        self.data_classes = [i.split('/')[-1] for i in data_class_paths]
        self.data_classes.sort()
        self.data_classes_idx = {label:idx for idx,label in enumerate(self.data_classes)}

        hr_height = 96
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])

        lr_transform = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.CenterCrop(224),
            transforms.Resize((hr_height // 2, hr_height // 2), Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1])])

        hr_transform = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.CenterCrop(224),
            transforms.Resize((hr_height,hr_height)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])

        data_list_hr = []
        data_list_lr = []

        for idx,labels in enumerate(self.data_classes):
            img_path = os.path.join(datapath,labels,'*')
            img_list = glob.glob(img_path)
            img_tensors_hr = [hr_transform(Image.open(imgs).convert("RGB")) for imgs in img_list]
            img_tensors_lr = [lr_transform(Image.open(imgs).convert("RGB")) for imgs in img_list]

            data_list_hr.append(torch.stack(img_tensors_hr))
            data_list_lr.append(torch.stack(img_tensors_lr))


        self.data_hr = torch.cat(data_list_hr,dim=0)
        self.data_lr = torch.cat(data_list_lr,dim=0)
        self.length = self.data_hr.shape[0]

    def __getitem__(self,index):
        return self.data_hr[index], self.data_lr[index]
        
    def __len__(self):
        return self.length


if __name__ == "__main__":
    batch_size = 64
    traindataset = read_covid_chest_xray(datapath='data/chest_xray/train')
    trainloader = DataLoader(traindataset, batch_size=batch_size, num_workers=24, shuffle=True)

    deivce = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    gen = Generator(in_channels=3).cuda()
    disc = Discriminator(in_channels=3).cuda()
    opt_gen = optim.Adam(gen.parameters(), lr=1e-2)
    opt_disc = optim.Adam(disc.parameters(), lr=1e-3)
    mse = nn.MSELoss()
    bce = nn.BCEWithLogitsLoss()
    vgg_loss = VGGLoss()

    generator_loss,discriminator_loss = train_fn(trainloader, disc, gen, opt_gen, opt_disc, mse, bce, vgg_loss, deivce, 30)