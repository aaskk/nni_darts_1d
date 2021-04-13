# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np
import torch
from torchvision import transforms
from torchvision.datasets import CIFAR10
import torch.utils.data as Data

class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask

        return img

def get_some(dataset,num):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    datas=np.array([])
    lables=np.zeros(num)
    for i in range(num):
        data,lable=dataset.data[i],dataset.targets[i]
        datas=np.append(datas,data)
        # datas[i]=data
        #datas=torch.stack((datas,data))
        lables[i]=lable
   # datas=np.asarray(datas)
   # lables=np.asarray(lables)
   # datas = torch.from_numpy(datas.astype(np.float32)).cuda(device)
    lables= torch.from_numpy(lables.astype(np.long)).cuda(device)
    datas_=datas.reshape(num,3,32,32)

    datas_ = torch.from_numpy(datas_.astype(np.float32)).cuda(device)
    dataset_=Data.TensorDataset(datas_, lables)
    return dataset_


def get_dataset(cls, cutout_length=0):
    MEAN = [0.49139968, 0.48215827, 0.44653124]
    STD = [0.24703233, 0.24348505, 0.26158768]
    transf = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip()
    ]
    normalize = [
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ]
    cutout = []
    if cutout_length > 0:
        cutout.append(Cutout(cutout_length))

    train_transform = transforms.Compose(transf+normalize +cutout)
    valid_transform = transforms.Compose(normalize)

    if cls == "cifar10":
        dataset_train = CIFAR10(root="./data", train=True, download=True, transform=train_transform)
        dataset_valid = CIFAR10(root="./data", train=False, download=True, transform=valid_transform)
        #data, lable = dataset_train.data[0],dataset_train.targets[0]
    else:
        raise NotImplementedError
    #dataset_train_, dataset_valid_=get_some(dataset_train,2000),get_some(dataset_valid,2000)
    dataset_train_, dataset_valid_=dataset_train, dataset_valid
    return dataset_train_, dataset_valid_
if __name__ == "__main__":


    get_dataset(cls='cifar10', cutout_length=0)