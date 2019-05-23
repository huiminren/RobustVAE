#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 13:45:23 2019

@author: huiminren

Reference: https://github.com/bhpfelix/Variational-Autoencoder-PyTorch/blob/master/src/vanila_vae.py
"""
from __future__ import print_function
import torch
import torch.utils.data as Data
from torch import nn,optim

import torchvision
import torchvision.transforms as transforms

from torchvision.utils import save_image

import numpy as np

import os
import glob
import sys
sys.path.append("..")
from utils import *

## custom weights initialization called on VAE_Net
#def weights_init(m):
#    classname = m.__class__.__name__
#    if classname.find('Conv') != -1:
#        m.weight.data.normal_(0.0, 0.02)
#    elif classname.find('BatchNorm') != -1:
#        m.weight.data.normal_(1.0, 0.02)
#        m.bias.data.fill_(0)
        
def random_noise(image,amount):
    out = image.copy()
    p = amount
    q = 0.5
    flipped = np.random.choice([True, False], size=image.shape,
                               p=[p, 1 - p])
    salted = np.random.choice([True, False], size=image.shape,
                              p=[q, 1 - q])
    peppered = ~salted
    out[flipped & salted] = 1
    out[flipped & peppered] = 0
    return out


class VAE_Net(nn.Module):
    def __init__(self, nc, ngf, ndf, nz):
        super(VAE_Net,self).__init__()
        ####################
        # ((H_in+2*padding-1*(kernel-1)-1)/stride)+1 
        ####################
        
        self.nc = nc
        self.ngf = ngf
        self.ndf = ndf
        self.nz = nz
        
        # encoder
        # input size is (nc) * 64 * 64
        self.e1 = nn.Conv2d(nc, ndf, 4, 2, 1)
        # ((64+2*1-1*(4-1)-1)/2)+1 = 32
        # state size. (ndf) * 32 * 32
        self.e2 = nn.Conv2d(ndf, ndf*2, 4, 2, 1)
        self.bn2 = nn.BatchNorm2d(ndf*2)
        # ((32+2*1-1*(4-1)-1)/2)+1 = 16
        # state size. (ndf*2) * 16 * 16
        self.e3 = nn.Conv2d(ndf*2, ndf*4, 4, 2, 1)
        self.bn3 = nn.BatchNorm2d(ndf*4)
        # ((16+2*1-1*(4-1)-1)/2)+1 = 8
        # state size. (ndf*4) * 8 * 8
        self.e4 = nn.Conv2d(ndf*4, ndf*8, 4, 2, 1)
        self.bn4 = nn.BatchNorm2d(ndf*8)
        # ((8+2*1-1*(4-1)-1)/2)+1 = 4
        # state size. (ndf*8) * 4 * 4
        self.fc1 = nn.Linear(ndf*8*4*4, nz)
        self.fc2 = nn.Linear(ndf*8*4*4, nz)
        
        # decoder
        # input size is nz
        self.d1 = nn.Linear(nz, ngf*8*4*4)
        # state size. will be reshape to ngf*8 x 4 x 4
        self.d2 = nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1)
        self.dbn2 = nn.BatchNorm2d(ngf * 4)
        # state size. (ngf*4) x 8 x 8
        self.d3 = nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1)
        self.dbn3 = nn.BatchNorm2d(ngf * 2)
        # state size. (ngf*4) x 16 x 16
        self.d4 = nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1)
        self.dbn4 = nn.BatchNorm2d(ngf)
        # state size. (ngf*2) x 32 x 32
        self.d5 = nn.ConvTranspose2d(ngf, nc, 4, 2, 1)
        self.dbn5 = nn.BatchNorm2d(nc)
        # state size. (nc) x 64 x 64
        
        
        self.leakyrelu = nn.LeakyReLU(0.2,inplace=True)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        
    
    def encoder(self,x):
        h1 = self.leakyrelu((self.e1(x))) # in_64, out_32
        h2 = self.leakyrelu(self.bn2(self.e2(h1))) # in_32, out_16
        h3 = self.leakyrelu(self.bn3(self.e3(h2))) # in_16, out_8
        h4 = self.leakyrelu(self.bn4(self.e4(h3))) # in_8, out_4
        h4 = h4.view(-1, self.ndf*8*4*4) # linear
        return self.fc1(h4), self.fc2(h4) # in_8192, out_nz
    
    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def decoder(self,z):
        h1 = self.relu(self.d1(z)) # in_nz, out_8192, out_4
        h1 = h1.view(-1, self.ngf*8, 4, 4) # reshape to conv2d, out_4
        h2 = self.relu(self.dbn2(self.d2(h1))) # in_4, out_8
        h3 = self.relu(self.dbn3(self.d3(h2))) # in_8, out_16
        h4 = self.relu(self.dbn4(self.d4(h3))) # in_16, out_32
        h5 = self.sigmoid(self.dbn5(self.d5(h4))) # in_32, out_64
        return h5
    
    def get_latent_var(self, x):
        mu, logvar = self.encoder(x.view(-1, self.nc, self.ndf, self.ngf))
        z = self.reparametrize(mu, logvar)
        return z

    def forward(self, x):
        mu, logvar = self.encoder(x.view(-1, self.nc, self.ndf, self.ngf))
        z = self.reparametrize(mu, logvar)
        res = self.decoder(z)
        return res, mu, logvar


class Autoencoder(object):
    
    def __init__(self):
        return 
    
    def loss_function(self,recon_x, x, mu, logvar):
        reconstruction_function = nn.BCELoss()
        reconstruction_function.size_average = False
        BCE = reconstruction_function(recon_x, x)
    
        # https://arxiv.org/abs/1312.6114 (Appendix B)
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
        KLD = torch.sum(KLD_element).mul_(-0.5)
#        print('loss',(BCE + KLD.item()))
#        print('bce',BCE.item())
#        print('kld',KLD.item())
        return BCE + KLD
    
    def train(self, device, model, train_loader, learning_rate, epochs):
        """
        """
        optimizer = optim.Adam(model.parameters(),lr=learning_rate, betas=(0.5, 0.999))
        model.train()
        for epoch in range(epochs):
            train_loss = 0
            for batch_idx, (data,) in enumerate(train_loader):
                data = data.to(device)
                
                # forward
                recon_batch, mu, logvar = model(data)
                loss = self.loss_function(recon_batch, data, mu, logvar)
                
                # backward
                optimizer.zero_grad()
                loss.backward()
                train_loss += loss.item()
                optimizer.step()
                
                if batch_idx % 100 == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader),
                        loss.item() / len(data)))
            
            if (epoch % 5 == 0) or (epoch == -1):
                print('====> Epoch: {} Average loss: {:.4f}'.format(
                  epoch, train_loss / len(train_loader.dataset)))
#                torch.save(model.state_dict(),'./models/Epoch_{}_Train_loss_{:.4f}.pth'.format(epoch, train_loss))
        return model
    
    def reconstruction(self, device, model, dataloader):
        model.eval()
        tmp=[]
        for batch_idx, (data,) in enumerate(dataloader):
            data = data.to(device)
            tmp.append(model(data)[0])
        recon = torch.cat(tmp, dim=0)
#        save_image(recon[:64],'recon.png')
        return recon
    
    def generation_eg(self, device, model,path):
        model.eval()
        z = torch.randn(10000, model.nz)
        z = z.to(device)
        gen = model.decoder(z)
        np.save(path+"generation_fid.npy",gen.detach().cpu().numpy())
        save_image(gen[:100].data, path+'genfid_eg.jpg',nrow=10, padding=2)
    
    def generation_fid(self, device, model,path):
        model.eval()
        z = torch.randn(10000, model.nz)
        z = z.to(device)
        gen = model.decoder(z)
        np.save(path+"generation_fid.npy",gen.detach().cpu().numpy())
        save_image(gen[:100].data, path+'genfid_eg.jpg',nrow=10, padding=2)
    
#======================================================================#
def main(debug=True):
    root = 'cnnvae_celcba/'
    if not os.path.isdir(root):
        os.mkdir(root)
    
    if len(sys.argv)>1:
        noise_factors = [float(sys.argv[1])]
    else:
#        noise_factors = [round(0.01*i,2) for i in range(1,52,4)]
        noise_factors = [0.2]
        
    
    ngf = 64
    ndf = 64
    nz = 100
    nc = 3
    
    batch_size = 128
    epochs = 100
    learning_rate = 1e-4
    use_cuda = True
    seed = 595
    
    if debug:
        epochs = 1

    #setting seed
    torch.manual_seed(seed) 
    
    # load cuda
    cuda = use_cuda and torch.cuda.is_available() 
    device = torch.device("cuda" if cuda else "cpu") #asign cuda
    
    data_files = glob.glob(os.path.join("../img_align_celeba", "*.jpg"))
    data_files = sorted(data_files)
    data_files = np.array(data_files)
    
    x_train = np.array([get_image(data_file, 148) for data_file in data_files])
    
    for noise_factor in noise_factors:
        x_train_noisy = random_noise(image = x_train, amount = noise_factor)    
        x_train_noisy = np.transpose(x_train_noisy,(0,3,1,2)).astype(np.float32) # change 2nd dimension as channel
        
        train_dataset = Data.TensorDataset(torch.tensor(x_train_noisy))
        train_loader = Data.DataLoader(train_dataset, batch_size=128, shuffle=True,num_workers=2)                         
        # load Neural Network
        net = VAE_Net(nc, ngf, ndf, nz)
        if torch.cuda.device_count() > 1: # if have multiple GPUs, set data parallel to model
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            net = nn.DataParallel(net)
        net.to(device)
        
        # train model
        vae = Autoencoder()
        model = vae.train(device = device, model = net, train_loader = train_loader, 
                         learning_rate = learning_rate, epochs = epochs)
        
        # get reconstruction
        recon_loader = Data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False,num_workers=2)
        vae.reconstruction(device=device, model=model, dataloader=recon_loader)
        
        # get generation
        vae.generation_fid(device=device, model=model, path="")


if __name__ == "__main__":
#    main(debug=True)
    ngf = 64
    ndf = 64
    nz = 100
    nc = 3
    
    batch_size = 128
    epochs = 100
    learning_rate = 1e-4
    use_cuda = True
    seed = 595
    
    epochs = 1

    #setting seed
    torch.manual_seed(seed) 
    
    # load cuda
    cuda = use_cuda and torch.cuda.is_available() 
    device = torch.device("cuda" if cuda else "cpu") #asign cuda
    
    data_files = glob.glob(os.path.join("../img_align_celeba", "*.jpg"))
    data_files = sorted(data_files)
    data_files = np.array(data_files)
    
    x_train = np.array([get_image(data_file, 148) for data_file in data_files])
    noise_factors=[0.05]
    for noise_factor in noise_factors:
        x_train_noisy = random_noise(image = x_train, amount = noise_factor)    
        x_train_noisy = np.transpose(x_train_noisy,(0,3,1,2)).astype(np.float32) # change 2nd dimension as channel
        
        train_dataset = Data.TensorDataset(torch.tensor(x_train_noisy))
        train_loader = Data.DataLoader(train_dataset, batch_size=128, shuffle=True,num_workers=2)                         
        # load Neural Network
        net = VAE_Net(nc, ngf, ndf, nz)
        if torch.cuda.device_count() > 1: # if have multiple GPUs, set data parallel to model
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            net = nn.DataParallel(net)
        net.to(device)
        
        # train model
        vae = Autoencoder()
        model = vae.train(device = device, model = net, train_loader = train_loader, 
                         learning_rate = learning_rate, epochs = epochs)
        
        # get reconstruction
        recon_loader = Data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False,num_workers=2)
        vae.reconstruction(device=device, model=model, dataloader=recon_loader)
        
        # get generation
        vae.generation_eg(device=device, model=model, path="")
