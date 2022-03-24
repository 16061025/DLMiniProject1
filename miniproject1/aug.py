import torch
from torchvision import transforms as tr
from torchvision.transforms import Compose
from PIL import Image
import numpy as np
import random


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
    for i in range(1,5):
        raw = unpickle("data_batch_" + str(i))
        print(raw.keys())
        imarray = []
        augmented = []
        for j in range(len(raw[b'labels'])):
            rs = raw[b'data'][j].reshape((32,32,3),order='F')
            #Image.fromarray(rs, mode="RGB").rotate(270).show()
            imarray.append(Image.fromarray(rs, mode="RGB").rotate(270))
            for i in range(5):
                augmented.append((pipeline(imarray[j]),raw[b'labels'][j]))
            augmented.append((imarray[j],raw[b'labels'][j]))


    a = random.randrange(len(augmented))
    augmented[a][0].show()
    print(augmented[a][1])


    trainvalsplit = 1
    split = int(trainvalsplit*len(augmented))

    augmented = np.array(augmented)
    print(np.asarray(augmented[0][0]).shape)
    print(augmented[:,1])
    trainx = torch.Tensor([np.asarray(i) for i in augmented[:split,0]])
    trainy = torch.Tensor(augmented[:split,1].tolist())
    trainds = TensorDataset(trainx,trainy)
    return trainds