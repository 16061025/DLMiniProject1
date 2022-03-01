import torch
import torch.nn as nn

from torch.utils.data import DataLoader

import torchvision.transforms as transforms
import torchvision.datasets as datasets

import netconfig
import numpy as np

class DataProcessor:

    def __init__(self):

        data_ROOT = netconfig.data_ROOT

        data_transforms = transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor(),
        ])

        self.test_data = datasets.CIFAR10(root=data_ROOT,
                                     train=False,
                                     download=False,
                                     transform=data_transforms)

        self.train_data = datasets.CIFAR10(root=data_ROOT,
                                      train=True,
                                      download=False,
                                      transform=data_transforms)

        batch_size = netconfig.batch_size

        self.train_dataloader = DataLoader(self.train_data, shuffle=True, batch_size=batch_size)
        self.test_dataloader = DataLoader(self.test_data, batch_size=batch_size)

        return self.train_dataloader, self.test_dataloader