# PyTorch StudioGAN: https://github.com/POSTECH-CVLab/PyTorch-StudioGAN
# The MIT License (MIT)
# See license file or visit https://github.com/POSTECH-CVLab/PyTorch-StudioGAN for details

# src/data_util.py

import os
import random

from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision.datasets import ImageFolder
from scipy import io
from PIL import ImageOps, Image
import torch
import torchvision.transforms as transforms
import h5py as h5
import numpy as np


class RandomCropLongEdge(object):
    """
    this code is borrowed from https://github.com/ajbrock/BigGAN-PyTorch
    MIT License
    Copyright (c) 2019 Andy Brock
    """
    """
    -----**********--------------
    |    **********    |        |
    |    **********    |        | size[0]=size[1]
    |    **********    |        |
    -----**********--------------
    """
    def __call__(self, img):
        # 获取长宽最小的一个
        size = (min(img.size), min(img.size))
        # Only step forward along this edge if it's the long edge
        # 随机取一个长边上的点，切一块方形
        i = (0 if size[0] == img.size[0] else np.random.randint(low=0, high=img.size[0] - size[0]))
        j = (0 if size[1] == img.size[1] else np.random.randint(low=0, high=img.size[1] - size[1]))
        return transforms.functional.crop(img, j, i, size[0], size[1])

    def __repr__(self):
        return self.__class__.__name__


class CenterCropLongEdge(object):
    """
    this code is borrowed from https://github.com/ajbrock/BigGAN-PyTorch
    MIT License
    Copyright (c) 2019 Andy Brock
    """
    """
    -----**********-----
    |    **********    |
    |    **********    | 
    |    **********    |        
    -----**********-----
    """
    def __call__(self, img):
        # 在整中心裁剪方形
        return transforms.functional.center_crop(img, min(img.size))

    def __repr__(self):
        return self.__class__.__name__


class Dataset_(Dataset):
    """
        数据集
    """
    def __init__(self,
                 data_name,
                 data_dir,
                 train,
                 crop_long_edge=False,
                 resize_size=None,
                 random_flip=False,
                 hdf5_path=None,
                 load_data_in_memory=False):
        super(Dataset_, self).__init__()
        # 数据集名称
        self.data_name = data_name
        # 数据集路径
        self.data_dir = data_dir
        # 是否是训练接
        self.train = train
        # 是否开启随机翻转
        self.random_flip = random_flip
        # hdf5的路径
        self.hdf5_path = hdf5_path
        # 是否将数据集加载到内存
        self.load_data_in_memory = load_data_in_memory
        # 变换集
        self.trsf_list = []
        # 如果hdf5路径为空
        if self.hdf5_path is None:
            # 如果开启长边裁剪
            if crop_long_edge:
                # 训练集采用随机裁剪，验证集采用中心裁剪
                crop_op = RandomCropLongEdge() if self.train else CenterCropLongEdge()
                # 添加裁剪操作
                self.trsf_list += [crop_op]
            # 如果开启resize操作
            if resize_size is not None:
                # 将resize添加到变换集
                self.trsf_list += [transforms.Resize(resize_size)]
        # 如果hdf5路径不为空
        else:
            # 变换集添加一个PIL图像转换
            self.trsf_list += [transforms.ToPILImage()]
        # 如果开启随机翻转
        if self.random_flip:
            # 添加一个随机的水平翻转变换
            self.trsf_list += [transforms.RandomHorizontalFlip()]
        # 添加张量转化以及标准化
        self.trsf_list += [transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
        # 整合所有变换
        self.trsf = transforms.Compose(self.trsf_list)
        # 加载数据集
        self.load_dataset()

    def load_dataset(self):
        # 如果有hdf5路径
        if self.hdf5_path is not None:
            # 打开hdf5文件
            with h5.File(self.hdf5_path, "r") as f:
                # 读取图像和标签
                data, labels = f["imgs"], f["labels"]
                # 获取数据集样例对个数
                self.num_dataset = data.shape[0]
                # 如果要加载到内存
                if self.load_data_in_memory:
                    print("Load {path} into memory.".format(path=self.hdf5_path))
                    # 用self把数据集保存在内存（利用类的性质）
                    self.data = data[:]
                    self.labels = labels[:]
            return
        # 没有hdf5
        # 如果数据集名称为CIFAR10
        if self.data_name == "CIFAR10":
            # 加载CIFAR10数据集
            self.data = CIFAR10(root=self.data_dir, train=self.train, download=True)
        # 如果数据集名称为CIFAR100
        elif self.data_name == "CIFAR100":
            # 加载CIFAR100数据集
            self.data = CIFAR100(root=self.data_dir, train=self.train, download=True)
        else:
            # 如果是训练集
            mode = "train" if self.train == True else "valid"
            # 获得训练集的数据集地址
            root = os.path.join(self.data_dir, mode)
            # 打开数据集
            self.data = ImageFolder(root=root)

    def _get_hdf5(self, index):
        # 打开hdf5
        with h5.File(self.hdf5_path, "r") as f:
            # 读取数据
            img = np.transpose(f["imgs"][index], (1, 2, 0))
            label = f["labels"][index]
        return img, label

    def __len__(self):
        # 没有hdf5
        if self.hdf5_path is None:
            # 数据集长度就是从本地图片文件夹打开的长度
            num_dataset = len(self.data)
        else:
            # 否则就是hdf5中数据集的长度
            num_dataset = self.num_dataset
        return num_dataset

    def __getitem__(self, index):
        # 没有hdf5
        if self.hdf5_path is None:
            # 数据集长度就是从本地图片文件夹打开的，从中获取一个数据
            img, label = self.data[index]
        # 有hdf5
        else:
            # 如果数据集在内存
            if self.load_data_in_memory:
                # 直接读
                img, label = np.transpose(self.data[index], (1, 2, 0)), self.labels[index]
            else:
                # 否则要去打开hdf5文件读
                img, label = self._get_hdf5(index)
        # 别忘了做变换
        return self.trsf(img), int(label)
