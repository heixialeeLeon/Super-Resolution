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
from transform.image_show import *
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize


# prepare the data
test_set = TrainDatasetFromFolder('/data_1/data/super-resolution/srgan/val', crop_size=256, upscale_factor=2)
test_loader = DataLoader(dataset=test_set, num_workers=4, batch_size=1, shuffle=False)

# prepare model
resume_model = "checkpoints_V1/netG_epoch_14.pth"
generator = Generator(16, 2)
generator.load_state_dict(torch.load(resume_model))
generator = generator.cuda()
generator = generator.eval()

def display_transform():
    return Compose([
        ToPILImage(),
        Resize(400),
        CenterCrop(400),
        ToTensor()
    ])


for item in test_loader:
    # CV2_showTensors(item[0])
    # CV2_showTensors(item[1])
    low_res = item[0]
    low_res = low_res.cuda()
    high_res = item[1]
    sr = generator(Variable(low_res))
    CV2_showTensors_Resize(low_res,sr,item[1],resize=(400,400),timeout=3000)
    #print(sr.shape)
    # low_rs = display_transform()(item[0].squeeze(0))
    # low_rs =low_rs.unsqueeze(0)
    # high_rs = display_transform()(item[1].squeeze(0))
    # high_rs = high_rs.unsqueeze(0)
    # sr =sr.cpu()
    # sr = display_transform()(sr.squeeze(0))
    # sr = sr.unsqueeze(0)
    # CV2_showTensors(low_rs, sr ,high_rs)
