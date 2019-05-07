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
from torchvision.transforms import ToTensor, ToPILImage
from model.model_v1 import Generator, Discriminator, FeatureExtractor
from dataset import TrainDatasetFromFolder
from transform.image_show import *
from PIL import Image
from torch.utils.data.dataset import Dataset
import torchvision.transforms.functional as tvF
from skimage import img_as_ubyte

# prepare model
resume_model = "checkpoints_V1/netG_epoch_14.pth"
generator = Generator(16, 2)
generator.load_state_dict(torch.load(resume_model))
generator = generator.cuda()
generator = generator.eval()

src_video_path = "video/ls.avi"
dst_video_path = "video/dst.avi"
dst_size = (640, 360)

videoCapture = cv2.VideoCapture(src_video_path)
if os.path.exists(dst_video_path):
    print(f"{dst_video_path} exist and remove now")
    os.remove(dst_video_path)

fps = videoCapture.get(cv2.CAP_PROP_FPS)

videoWriter = cv2.VideoWriter(dst_video_path, cv2.VideoWriter_fourcc('M', 'P', 'E', 'G'), fps, dst_size)
while (videoCapture.isOpened()):
    ret, frame = videoCapture.read()
    if ret == True:
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        img = Variable(ToTensor()(frame)).unsqueeze(0)
        img = img.cuda()
        out = generator(img)
        out_img = torch.squeeze(out).data.cpu().numpy().transpose(1, 2, 0)
        out_img = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
        out_img = out_img *255
        out_img = out_img.astype(np.uint8)
        # cv2.imshow("test",out_img)
        # cv2.waitKey(1000)
        videoWriter.write(out_img)
    else:
        break
videoCapture.release()
videoWriter.release()

# while(videoCapture.isOpened()):
#     ret, frame = videoCapture.read()
#     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     img = Variable(ToTensor()(frame)).unsqueeze(0)
#     img = img.cuda()
#     out = generator(img)
#     out = out.cpu()
#     out_img = torch.squeeze(out).data.cpu().numpy().transpose(1, 2, 0)
#     out_img = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
#     cv2.imshow("test",out_img)
#     cv2.waitKey(1000)