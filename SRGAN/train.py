import argparse
import os
from math import log10

import pandas as pd
import torch.optim as optim
import torch.utils.data
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from loss import GeneratorLoss
from model.model_v2 import Generator, Discriminator
from dataset import TrainDatasetFromFolder

parser = argparse.ArgumentParser(description='Train Super Resolution Models')

# training parameters
parser.add_argument("--image_size", type=int, default=256,help="training patch size")
parser.add_argument("--batch_size", type=int, default=4,help="batch size")
parser.add_argument("--epochs", type=int, default=60,help="number of epochs")
parser.add_argument("--save_per_epoch", type=int, default=5,help="number of epochs")
parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
parser.add_argument("--steps_show", type=int, default=100,help="steps per epoch")
parser.add_argument("--scheduler_step", type=int, default=10,help="scheduler_step for epoch")
parser.add_argument("--weight", type=str, default=None,help="weight file for restart")
parser.add_argument("--output_path", type=str, default="checkpoints",help="checkpoint dir")
parser.add_argument("--devices", type=str, default="cuda:0",help="device description")
parser.add_argument("--image_channels", type=int, default=3,help="batch image_channels")
parser.add_argument("--resume_model", type=str, default=None, help="resume model path")

args = parser.parse_args()
print(args)

def save_model_as(model, model_name):
    ckpt_name = '/'+model_name
    path = args.output_path
    if not os.path.exists(path):
        os.mkdir(path)
    path_final = path + ckpt_name
    print('Saving checkpoint to: {}\n'.format(path_final))
    torch.save(model.state_dict(), path_final)

def save_model(model,epoch):
    '''save model for eval'''
    ckpt_name = '/reid_epoch_{}.pth'.format(epoch)
    path = args.output_path
    if not os.path.exists(path):
        os.mkdir(path)
    path_final = path + ckpt_name
    print('Saving checkpoint to: {}\n'.format(path_final))
    torch.save(model.state_dict(), path_final)

def resume_model(model, model_path):
    print("Resume model from {}".format(args.resume_model))
    model.load_state_dict(torch.load(model_path))

def model_to_device(model):
    device = torch.device(args.devices if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return model

def tensor_to_device(tensor):
    device = torch.device(args.devices if torch.cuda.is_available() else "cpu")
    return tensor.to(device)

# prepare the data
train_set = TrainDatasetFromFolder('/data_1/data/super-resolution/srgan/train', crop_size=256, upscale_factor=2)
train_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=args.batch_size, shuffle=True)

# prepare the model
netG = Generator(2)
netG = model_to_device(netG)
print('# generator parameters:', sum(param.numel() for param in netG.parameters()))
netD = Discriminator()
netD = model_to_device(netD)
print('# discriminator parameters:', sum(param.numel() for param in netD.parameters()))

# prepare the criterion
generator_criterion = GeneratorLoss()

# prepare the optim
optimizerG = optim.Adam(netG.parameters())
optimizerD = optim.Adam(netD.parameters())

for epoch in range(args.epochs):

    netG.train()
    netD.train()
    for index, (data, target) in enumerate(train_loader):
        batch_size = data.size(0)
        ############################
        # (1) Update D network: maximize D(x)-1-D(G(z))
        ###########################
        real_img = Variable(tensor_to_device(target))
        z = Variable(tensor_to_device(data))
        fake_img = netG(z)

        netD.zero_grad()
        real_out = netD(real_img).mean()
        fake_out = netD(fake_img).mean()
        d_loss = 1-real_out+fake_out
        d_loss.backward(retain_graph=True)
        optimizerD.step()

        ############################
        # (2) Update G network: minimize 1-D(G(z)) + Perception Loss + Image Loss + TV Loss
        ###########################
        netG.zero_grad()
        g_loss = generator_criterion(fake_out, fake_img, real_img)
        g_loss.backward()
        optimizerG.step()

        if index % args.steps_show == 0:
            print("{}/{} d_loss: {}, real_out: {}, fake_out: {}, g_loss: {}".format(index, len(train_loader), d_loss.item(),real_out.item(),fake_out.item(),g_loss.item()))

    if (epoch + 1) % args.save_per_epoch == 0:
        g_model_name = "netG_epoch_{}.pth".format(epoch)
        d_model_name = "netD_epoch_{}.pth".format(epoch)
        save_model_as(netG, g_model_name)
        save_model_as(netD, d_model_name)