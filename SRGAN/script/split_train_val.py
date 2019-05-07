import os
import shutil

source_folder = "/data_1/data/VOC2007/VOCdevkit/VOC2007/JPEGImages"
rate = 0.7
target_folder = "/data_1/data/super-resolution/srgan"

train_dir = os.path.join(target_folder,"train")
val_dir = os.path.join(target_folder,"val")

if not os.path.exists(train_dir):
    os.mkdir(train_dir)

if not os.path.exists(val_dir):
    os.mkdir(val_dir)

file_list = os.listdir(source_folder)
train_length = int(len(file_list)*rate)

print(f"total file: {len(file_list)}, train length: {train_length}")

print("Copy the train files ...")
for item in file_list[:train_length]:
    src_file = os.path.join(source_folder,item)
    target_file = os.path.join(train_dir,item)
    shutil.copyfile(src_file, target_file)

print("Copy the val files ...")
for item in file_list[train_length:]:
    src_file = os.path.join(source_folder,item)
    target_file = os.path.join(val_dir,item)
    shutil.copyfile(src_file, target_file)

print("finished copy")