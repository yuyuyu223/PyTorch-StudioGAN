'''
Date: 2021-11-30 17:14:41
LastEditors: HowsenFisher
LastEditTime: 2021-12-01 12:58:41
FilePath: \PyTorch-StudioGAN\src\utils\hdf5.py
'''
"""
this code is borrowed from https://github.com/ajbrock/BigGAN-PyTorch

MIT License

Copyright (c) 2019 Andy Brock
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from os.path import dirname, exists, join, isfile
import os

from torch.utils.data import DataLoader
from tqdm import tqdm
import h5py as h5

from data_util import Dataset_


def make_hdf5(name, img_size, crop_long_edge, resize_size, data_dir, DATA, RUN):
    # hdf5文件名
    file_name = "{dataset_name}_{size}_train.hdf5".format(dataset_name=name, size=img_size)
    # 保存位置
    file_path = join(data_dir, file_name)
    # 保存文件夹
    hdf5_dir = dirname(file_path)
    # 如果文件夹不存在
    if not exists(hdf5_dir):
        # 创建文件夹
        os.makedirs(hdf5_dir)
    # 如果file_path有文件了
    if os.path.isfile(file_path):
        # 打印文件存在的信息
        print("{file_name} exist!\nThe file are located in the {file_path}.".format(file_name=file_name,
                                                                                    file_path=file_path))
    # 文件不存在
    else:
        # 读取数据集
        dataset = Dataset_(data_name=DATA.name,
                           data_dir=RUN.data_dir,
                           train=True,
                           crop_long_edge=crop_long_edge,
                           resize_size=resize_size,
                           random_flip=False,
                           hdf5_path=None,
                           load_data_in_memory=False)
        # 制作dataloader
        dataloader = DataLoader(dataset,
                                batch_size=500,
                                shuffle=False,
                                pin_memory=False,
                                num_workers=RUN.num_workers,
                                drop_last=False)

        print("Start to load {name} into an HDF5 file with chunk size 500.".format(name=name))
        for i, (x, y) in enumerate(tqdm(dataloader)):
            # [-1,1] --> [0,255]
            x = (255 * ((x + 1) / 2.0)).byte().numpy() # [500,3,img_size,img_size]
            y = y.numpy()
            if i == 0:
                with h5.File(file_path, "w") as f:
                    print("Produce dataset of len {num_dataset}".format(num_dataset=len(dataset)))
                    # 开辟了一些空间去存x
                    imgs_dset = f.create_dataset("imgs",
                                                 x.shape,
                                                 dtype="uint8",
                                                 maxshape=(len(dataset), 3, img_size, img_size),
                                                 chunks=(500, 3, img_size, img_size),
                                                 compression=False)
                    print("Image chunks chosen as {chunk}".format(chunk=str(imgs_dset.chunks)))
                    imgs_dset[...] = x
                    # 开辟了一些空间去存y
                    labels_dset = f.create_dataset("labels",
                                                   y.shape,
                                                   dtype="int64",
                                                   maxshape=(len(dataloader.dataset), ),
                                                   chunks=(500, ),
                                                   compression=False)
                    print("Label chunks chosen as {chunk}".format(chunk=str(labels_dset.chunks)))
                    labels_dset[...] = y
            else:
                with h5.File(file_path, "a") as f:
                    # resize可以在当前内存基础上，再开辟出一块长度为x.shape[0]大小的内存，填充应该为0
                    f["imgs"].resize(f["imgs"].shape[0] + x.shape[0], axis=0)
                    # 新开辟的这块内存赋值
                    f["imgs"][-x.shape[0]:] = x
                    # y同理
                    f["labels"].resize(f["labels"].shape[0] + y.shape[0], axis=0)
                    f["labels"][-y.shape[0]:] = y
    return file_path, False, None
