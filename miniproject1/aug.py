import torch
from torchvision import transforms as tr
from torchvision.transforms import Compose
from PIL import Image
import numpy as np
import random
import netconfig
from torch.utils.data import TensorDataset

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


pipeline = Compose([
    tr.RandomAffine(10, translate=(1 / 8, 1 / 8), shear=10),
    tr.RandomHorizontalFlip(0.5),

])
def augData():
    augmented_samples = []
    augmented_labels = []
    for i in range(1,6):
        raw = unpickle(netconfig.data_ROOT+"/cifar-10-batches-py/data_batch_" + str(i))
        for j in range(len(raw[b'labels'])):

            origin_sample = np.array(raw[b'data'][j]).reshape(3,32,32)/255
            origin_label = np.array(raw[b'labels'][j]).astype(int)
            augmented_samples.append(origin_sample)
            augmented_labels.append(origin_label)

            tmp = origin_sample.reshape((32,32,3),order='F')
            rotate_img = Image.fromarray(tmp, mode="RGB").rotate(270)
            rotate_sample = np.array(rotate_img).reshape((3,32,32))/255
            print(rotate_sample)
            augmented_samples.append(rotate_sample)
            augmented_labels.append(origin_label)

            for k in range(0,2):
                new_sample = np.array(pipeline(rotate_img)).reshape((3,32,32))/255
                print(new_sample)
                augmented_samples.append(new_sample)
                augmented_labels.append(origin_label)
    print(augmented_samples[0])
    augmented_samples = np.array(augmented_samples)
    augmented_labels = np.array(augmented_labels)
    s = torch.Tensor(augmented_samples).float()
    l = torch.Tensor(augmented_labels).long()
    augmented_dataset = TensorDataset(s, l)
    return augmented_dataset
