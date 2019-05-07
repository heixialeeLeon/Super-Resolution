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

# prepare model
resume_model = "checkpoints_V1/netG_epoch_9.pth"
generator = Generator(16, 2)
generator.load_state_dict(torch.load(resume_model,map_location='cuda:0'))
generator = generator.cuda()
generator = generator.eval()

data_root = "/data_1/data/super-resolution/0203"
image_list = os.listdir(data_root)

def test_transform(crop_size):
    return Compose([
        CenterCrop(crop_size),
        ToTensor(),
    ])

class TrainDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, crop_size):
        super(TrainDatasetFromFolder, self).__init__()
        self.image_filenames = [os.path.join(dataset_dir, x) for x in os.listdir(dataset_dir)]
        self.transform = test_transform(crop_size)

    def __getitem__(self, index):
        img = self.transform(Image.open(self.image_filenames[index]))
        return img

    def __len__(self):
        return len(self.image_filenames)

test_set = TrainDatasetFromFolder(data_root, crop_size=256)
test_loader = DataLoader(dataset=test_set, num_workers=4, batch_size=1, shuffle=False)

for item in test_loader:
    item = item.cuda()
    sr = generator(Variable(item))
    CV2_showTensors_Resize(item, sr ,resize=(512,512),timeout=0)
