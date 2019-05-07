import argparse
import os
import sys

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn as nn
from torch.autograd import Variable

import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from model.model_v1 import Generator, Discriminator, FeatureExtractor
from dataset import TrainDatasetFromFolder

parser = argparse.ArgumentParser(description='Train Super Resolution Models')

# training parameters
# parser.add_argument("--image_size", type=int, default=256,help="training patch size")
parser.add_argument("--data_dir", type=str, default="/data_1/data/super-resolution/srgan/train",help="data dir location")
parser.add_argument("--batch_size", type=int, default=4,help="batch size")
parser.add_argument("--epochs", type=int, default=60,help="number of epochs")
parser.add_argument("--save_per_epoch", type=int, default=5,help="number of epochs")
parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
parser.add_argument('--generatorLR', type=float, default=0.0001, help='learning rate for generator')
parser.add_argument('--discriminatorLR', type=float, default=0.0001, help='learning rate for discriminator')
parser.add_argument("--steps_show", type=int, default=100,help="steps per epoch")
parser.add_argument("--scheduler_step", type=int, default=10,help="scheduler_step for epoch")
parser.add_argument("--weight", type=str, default=None,help="weight file for restart")
parser.add_argument("--output_path", type=str, default="checkpoints_V1",help="checkpoint dir")
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
train_set = TrainDatasetFromFolder(args.data_dir, crop_size=256, upscale_factor=2)
train_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=args.batch_size, shuffle=True)

# prepare the model
generator = Generator(16, 2)
generator = model_to_device(generator)
discriminator = Discriminator()
discriminator = model_to_device(discriminator)

# prepare the loss
feature_extractor = FeatureExtractor(torchvision.models.vgg19(pretrained=True))
feature_extractor = model_to_device(feature_extractor)
content_criterion = nn.MSELoss()
adversarial_criterion = nn.BCELoss()
# ones_const = Variable(torch.ones(args.bache_size, 1))

# prepare the optim
optim_generator = optim.Adam(generator.parameters(), lr=args.generatorLR)
optim_discriminator = optim.Adam(discriminator.parameters(), lr=args.discriminatorLR)

#generator pre-train
print("start the generator pre-train ...")
# low_res = torch.FloatTensor(args.batch_size, 3, args.image_size, args.image_size)
for epoch in range(2):
    mean_generator_content_loss = 0.0
    for i, (data, target) in enumerate(train_loader):

        high_res_real = Variable(tensor_to_device(target))
        high_res_fake = generator(Variable(tensor_to_device(data)))

        generator.zero_grad()
        generator_content_loss = content_criterion(high_res_fake, high_res_real)
        mean_generator_content_loss += generator_content_loss.item()
        generator_content_loss.backward()
        optim_generator.step()
        if i % args.steps_show == 0:
            print("epoch {} {}/{} current loss : {}".format(epoch, i, len(train_loader), generator_content_loss.item()))

save_model_as(generator,'generator_pretrain.pth')

# SRGAN training
print("start the SRGAN training ...")
optim_generator = optim.Adam(generator.parameters(), lr=args.generatorLR*0.1)
optim_discriminator = optim.Adam(discriminator.parameters(), lr=args.discriminatorLR*0.1)

for epoch in range(args.epochs):
    mean_generator_content_loss = 0.0
    mean_generator_adversarial_loss = 0.0
    mean_generator_total_loss = 0.0
    mean_discriminator_loss = 0.0

    for i, (data, target) in enumerate(train_loader):
        batch_size = data.size(0)

        high_res_real = Variable(tensor_to_device(target))
        high_res_fake = generator(Variable(tensor_to_device(data)))
        target_real = Variable(torch.rand(batch_size,1)*0.5+0.7).cuda()
        target_fake = Variable(torch.rand(batch_size,1)*0.3).cuda()

        ######### Train discriminator #########
        discriminator.zero_grad()
        discriminator_loss = adversarial_criterion(discriminator(high_res_real), target_real) + \
                             adversarial_criterion(discriminator(Variable(high_res_fake.data)), target_fake)
        mean_discriminator_loss += discriminator_loss.item()
        discriminator_loss.backward()
        optim_discriminator.step()

        ######### Train generator #########
        generator.zero_grad()
        real_features = Variable(feature_extractor(high_res_real).data)
        fake_features = feature_extractor(high_res_fake)

        generator_content_loss = content_criterion(high_res_fake, high_res_real) + 0.006 * content_criterion(fake_features, real_features)
        mean_generator_content_loss += generator_content_loss.item()

        ones_const = Variable(torch.ones(batch_size, 1).cuda())
        generator_adversarial_loss = adversarial_criterion(discriminator(high_res_fake), ones_const)
        mean_generator_adversarial_loss += generator_adversarial_loss.item()

        generator_total_loss = generator_content_loss + 1e-3 * generator_adversarial_loss
        mean_generator_total_loss += generator_total_loss.item()

        generator_total_loss.backward()
        optim_generator.step()

        if i % args.steps_show == 0:
            print("epoch{} {}/{} Discriminator_Loss: {}, Generator_Loss: {}, {}, {}".format(epoch, i, len(train_loader), discriminator_loss.item(),
                                                                        generator_content_loss.item(),generator_adversarial_loss.item(), generator_total_loss))

    print("******************************************************************************************")
    print("{} epoch: discriminator_loss: {}".format(epoch,mean_discriminator_loss/len(train_loader)))
    print("{} epoch: generator_content_loss: {}".format(epoch, mean_generator_content_loss / len(train_loader)))
    print("{} epoch: generator_adversarial_loss: {}".format(epoch, mean_generator_adversarial_loss / len(train_loader)))
    print("{} epoch: generator_total_loss: {}".format(epoch, mean_generator_total_loss / len(train_loader)))
    print("******************************************************************************************")

    if (epoch + 1) % args.save_per_epoch == 0:
        g_model_name = "netG_epoch_{}.pth".format(epoch)
        d_model_name = "netD_epoch_{}.pth".format(epoch)
        save_model_as(generator, g_model_name)
        save_model_as(discriminator, d_model_name)