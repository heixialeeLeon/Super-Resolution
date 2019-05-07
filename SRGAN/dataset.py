from os import listdir
from os.path import join

from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize
from transform.image_show import *

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])

def train_hr_transform(crop_size):
    return Compose([
        CenterCrop(crop_size),
        #Resize(size=(crop_size,crop_size),interpolation=Image.BICUBIC),
        ToTensor(),
    ])


def train_lr_transform(crop_size, upscale_factor):
    return Compose([
        ToPILImage(),
        Resize(crop_size // upscale_factor, interpolation=Image.BICUBIC),
        ToTensor()
    ])

class TrainDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, crop_size, upscale_factor):
        super(TrainDatasetFromFolder, self).__init__()
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]
        # crop_size = calculate_valid_crop_size(crop_size, upscale_factor)
        self.hr_transform = train_hr_transform(crop_size)
        self.lr_transform = train_lr_transform(crop_size, upscale_factor)

    def __getitem__(self, index):
        hr_image = self.hr_transform(Image.open(self.image_filenames[index]))
        lr_image = self.lr_transform(hr_image)
        #print(self.image_filenames[index])
        return lr_image, hr_image

    def __len__(self):
        return len(self.image_filenames)


if __name__ == "__main__":
    train_set = TrainDatasetFromFolder('/data_1/data/super-resolution/srgan/train', crop_size=256, upscale_factor=2)
    for item in train_set:
        CV2_showTensors(item[0])
        CV2_showTensors(item[1])
        # print(np.max(item[0].numpy()))
        # print(np.max(item[1].numpy()))
