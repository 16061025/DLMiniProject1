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

    # layer 2 3 ...
    for i in range(1, N):
        width_out = np.floor((width_in + 2 - F[i]) / 2 + 1)
        #print('width_out', width_out)
        skip_kernel_size = max(width_in + 2 - (width_out * 2 + 1), 1)
        K.append(int(skip_kernel_size))
        width_in = width_out

    P = int(width_out)

    print('K=',K)
    print('P=',P)

    return K, P

N = 4  # Residual Layers

B = [3, 2, 3, 2]  # Residual blocks in Residual Layer i

C1 = 16

C = C1 * (2**np.arange(0, N, 1))  # channels in Residual Layer i

F = [3, 5, 5, 3]  # Conv. kernel size in Residual Layer i

# K Skip connection kernel size in Residual Layer i
# P Average pool kernel size
K, P = NBF2KP(N, B, F)


def setconfig(config):
    global N, B, C1, C, F, K, P
    N = config['N']
    B = config['B']
    C1= config['C1']
    F = config['F']

    C = C1 * (2**np.arange(0, N, 1))
    K, P = NBF2KP(N, B, F)



# parameter size estimate
#math
# before net
# 29C0
#
# total direct conv
# sum (2Bi-0.5)Ci^2Fi^2 + 0.5C0^2F0^2
#
# total direct nor
# sum 4BiFi
# total skip conv and nor
# sum (1:N) (Ci-1CiKi^2 + 2Ci)

# def my_count_para():
#     total = 0
#
#     total += 3 * netconfig.C[0] * (3 * 3)
#     total += netconfig.C[0] * 2
#
#     for i in range(0, netconfig.N):
#         total += netconfig.B[i] * 2 * netconfig.C[i] * netconfig.C[i] * (netconfig.F[i] ** 2)
#         total += netconfig.B[i] * ((netconfig.C[i] * 2) * 2)
#
#         if i != 0:
#             total-= 0.5 * netconfig.C[i] * netconfig.C[i] * (netconfig.F[i] ** 2)
#             total += netconfig.C[i] * netconfig.C[i-1] * (netconfig.K[i]**2)
#             total += netconfig.C[i] * 2
#
#     total+=(netconfig.C[-1]+1) * 10
#     return total
#
# print("my number of parameters", my_count_para())



