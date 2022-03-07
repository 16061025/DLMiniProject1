import numpy as np
#data set path
data_ROOT = "D:/dataset"

# runtime device
device = "cpu"

# train batch size
batch_size = 64

# learning rate
lr = 0.001

# model hyper-parameters

N = 4  # Residual Layers

B = [2, 2, 2, 2]  # Residual blocks in Residual Layer i

C1 = 64

C = C1 * (2**np.arange(0, N, 1))  # channels in Residual Layer i

F = [3, 3, 3, 3]  # Conv. kernel size in Residual Layer i

K = [1, 1, 1, 1]  # Skip connection kernel size in Residual Layer i

P = 4   # Average pool kernel size

# parameter size estimate



