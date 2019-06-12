from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import argparse
from dataset import load_img_align_celeba

class Gen(nn.Module):
    def __init__(self, ngpu):
        super(Gen, self).__init__()
        self.ngpu = ngpu

        def block(in_feat, out_feat):
            layers = [nn.Linear(in_feat, out_feat)]
            layers.append(nn.ReLU())
            return layers
        self.main = nn.Sequential(
            *block(100, 256),
            *block(256, 512),
            nn.Linear(512, 784), 
            nn.Sigmoid())
    def forward(self, input):
        return self.main(input)

class Dis(nn.Module):
    def __init__(self, ngpu):
        super(Dis, self).__init__()
        self.ngpu = ngpu

        def block(in_feat, out_feat):
            layers = [nn.Linear(in_feat, out_feat)]
            layers.append(nn.LeakyReLU(0.2))
            return layers

        self.main = nn.Sequential(
            *block(784, 256),
            *block(256, 256),
            nn.Linear(256, 1))
            
    def forward(self, input):
        return self.main(input)

def showloss(G_losses, D_losses, result_dir):
    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses,label="G")
    plt.plot(D_losses,label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(result_dir + '/Loss.png')

def imshow(imgs, result_dir):
    imgs = torchvision.utils.make_grid(imgs, pad_value=0)
    npimgs = imgs.numpy()
    plt.figure(figsize=(8, 8))
    plt.imshow(np.transpose(npimgs, (1, 2, 0)))
    plt.xticks([])
    plt.yticks([])
    plt.savefig(result_dir + '/LastStep.png')

def salt_and_pepper(imgs, device, p=0.29):
    #imgs.to('cpu')
    """salt and pepper noise for mnist"""
    result = imgs[:]
    for i, img in enumerate(imgs):
        idx = torch.randperm(img.view(-1,).size(0))[:int(img.view(-1,).size(0)*p)]
        noisy = img.view(-1)[:]
        original = img.view(-1)[:]
        noisy = torch.where(noisy<=0.5, torch.ones(noisy.size()).to(device), torch.zeros(noisy.size()).to(device))
        original[idx] = noisy[idx]
        result[i, :] = original.view(img.size())
    return result

def main(noise_factor, data, gan_model):

############################
    result_dir = './result/'+gan_model+data+str(noise_factor)
    BATCH_SIZE=64
    WORKERS = 2
    NGPU = 1

    Z_dim = 100
    X_dim = 784
    Img_dim = 28

    LR = 0.0002
    N_EPOCHS = 200
###########################


    transform = transforms.Compose(
                [transforms.ToTensor()])
#                transforms.Normalize([0.5], [0.5])])

    dataset_class = getattr(torchvision.datasets, data)
    trainset = dataset_class(root='./data', train=True, 
                            download=True, transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                              shuffle=True, num_workers=WORKERS)


    # Decide which device we want to run on
    device = torch.device("cuda:0" if (torch.cuda.is_available() and NGPU > 0) else "cpu")

    netG = Gen(NGPU).to(device)
    netD = Dis(NGPU).to(device)
    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (NGPU > 1):
        netG = nn.DataParallel(netG, list(range(NGPU)))
        netD = nn.DataParallel(netD, list(range(NGPU)))

    print(netG)
    print(netD)
    print(device)

    criterion = nn.BCEWithLogitsLoss()
    sig = nn.Sigmoid()
    # Create batch of latent vectors that we will use to visualize
    # the result of the generator
    fixed_noise = torch.randn(64, Z_dim, device=device)
    # Establish convention for real and fake labels
    real_label = 1
    fake_label = 0

    # Setup Adam optimizers
    optimizerD = optim.Adam(netD.parameters(), lr=LR) 
    optimizerG = optim.Adam(netG.parameters(), lr=LR)

    # Training Loop

    # Lists to keep track of progress
    G_losses = []
    D_losses = []

    # results save folder
    if not os.path.isdir(result_dir):
        os.mkdir(result_dir)

    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(N_EPOCHS):
        # For each batch
        for i, data in enumerate(trainloader):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            netD.zero_grad()
            # Format batch
            real = data[0].to(device)
            b_size = real.size(0)
            label = torch.full((b_size,), real_label, device=device)
            # Forward pass real batch through D
            real = real.view(-1, X_dim)
            real = salt_and_pepper(real, device, p=noise_factor).to(device)
            output = netD(real).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            output = sig(output)
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, Z_dim, device=device)
            # Generate fake image batch with G
            fake = netG(noise)
            label.fill_(fake_label)
            # Classify all fake batch with D
            # Detach to avoid training G on these labels (&save time)
            output = netD(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch
            errD_fake.backward()
            output = sig(output)
            D_G_z1 = output.mean().item()
            # Add the gradients from the all-real and all-fake batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            output = sig(output)
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()

            # Output training stats
            if i % 200 == 0:
                print(f'[{epoch}/{N_EPOCHS}], {i}, {len(trainloader)}, Loss_D: {errD.item()}, '
                        f'Loss_G: {errG.item()}, D(x): {D_x}, D(G(z)): {D_G_z1}/{D_G_z2}')

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if i == len(trainloader)-1:
            # if epoch == N_EPOCHS-1 and i == len(trainloader)-1:
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                    np.save(result_dir+'/'+str(epoch), fake.numpy())

    showloss(G_losses, D_losses, result_dir)
    imshow(torch.reshape(fake, (64, 1, Img_dim, Img_dim)), result_dir) 
    # result for FID
    z_ = torch.randn(10000, Z_dim, device=device)
    z_fid = netG(z_).detach().cpu()
    np.save(result_dir + '/result4FID', z_fid.numpy())

if __name__ == '__main__':
    # parse the parameters to run from terminal arguments
    parser = argparse.ArgumentParser(
                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--noisefactor', type=float, default=0., help='the noise factor for salt and pepper')
    parser.add_argument('--taskid', type=int, default=0, help='the experiment task to run')
    args = parser.parse_args()

    noise_factor = args.noisefactor
    t = args.taskid
    # Basic GAN and MNIST
    if t == 0:
        data = 'MNIST'
        gan_model = 'basicGAN'

    if t == 1:
        data = 'FashionMNIST'
        gan_model = 'basicGAN'

    main(noise_factor, data, gan_model)
