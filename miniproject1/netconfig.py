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

def NBF2KP(N, B, F):
    width_in = 32

    K = []

    # layer 1
    K.append(2*F[0] - 5)
    width_in = width_in + B[0] * 2 * (3 - F[0])
    print('width_out', width_in)

    # layer 2 3 ...
    for i in range(1, N):
        width_out = np.floor((width_in + 2 - F[i]) / 2 + 1)
        print('width_out', width_out)
        skip_kernel_size = max(width_in + 2 - (width_out * 2 + 1), 1)
        K.append(int(skip_kernel_size))
        width_in = width_out

        # width_out_after_first_block = np.floor((width_in + 2 - F[i]) / 2 + 1) + (3 - F[i])
        # print('width_out_after_first_block', width_out_after_first_block)
        # skip_kernel_size = max(width_in + 2 - (width_out_after_first_block * 2 + 1), 1)
        # K.append(int(skip_kernel_size))
        # width_out_after_layer = width_out_after_first_block + (B[i]-1) * 2 * (3 - F[i])
        # width_in = width_out_after_layer


    P = int(width_out)

    print(K, P)

    return K, P

N = 5  # Residual Layers

B = [3, 2, 3, 2, 2]  # Residual blocks in Residual Layer i

C1 = 64

C = C1 * (2**np.arange(0, N, 1))  # channels in Residual Layer i

F = [3, 5, 5, 3, 3]  # Conv. kernel size in Residual Layer i


# K Skip connection kernel size in Residual Layer i
# P Average pool kernel size
K, P = NBF2KP(N, B, F)


# parameter size estimate



