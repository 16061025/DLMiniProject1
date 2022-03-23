import numpy as np

# model hyper-parameters
def NBF2KP(N, B, F):
    width_in = 32

    K = []

    # layer 1
    K.append(2*F[0] - 5)
    width_in = width_in #+ B[0] * 2 * (3 - F[0])
    #print("width_out", width_in)

    # layer 2 3 ...
    for i in range(1, N):
        width_out = np.floor((width_in + 2 - F[i]) / 2 + 1)
        #print('width_out', width_out)
        skip_kernel_size = max(width_in + 2 - (width_out * 2 + 1), 1)
        K.append(int(skip_kernel_size))
        width_in = width_out

    P = int(width_out)

    #print('K=',K)
    #print('P=',P)

    return K, P


def setconfig(config):
    global N, B, C1, C, F, K, P
    N = config['N']
    B = config['B']
    C1= config['C1']
    C = C1 * (2 ** np.arange(0, N, 1))
    F = config['F']
    K = config['K']
    P = config['P']

    print('N=', N)
    print('B=', B)
    print('C1=', C1)
    print('F=', F)
    print('K=', K)
    print('P=', P)



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

def my_count_para(N, C, B, F, K):
    total = 0

    total += 3 * C[0] * (3 * 3)
    total += C[0] * 2

    for i in range(0, N):
        total += B[i] * 2 * C[i] * C[i] * (F[i] ** 2)
        total += B[i] * ((C[i] * 2) * 2)

        if i != 0:
            total-= 0.5 * C[i] * C[i] * (F[i] ** 2)
            total += C[i] * C[i-1] * (K[i]**2)
            total += C[i] * 2

    total+=(C[-1]+1) * 10
    return total

# print("my number of parameters", my_count_para())

def verifyPara(N, C, B, F, K):
    width_in = 32
    for i in range(1, N):
        width_out = np.floor((width_in + 2 - F[i]) / 2 + 1)
        if width_out <= 0:
            return False
        width_in = width_out
    para_cnt = my_count_para(N, C, B, F, K)
    if para_cnt > 5000000 or para_cnt <4500000:
        return False
    print("para count", para_cnt)
    return True

#data set path
data_ROOT = "D:/dataset"

# runtime device
device = "cpu"

# train batch size
batch_size = 64

# learning rate
lr = 0.001

N = 6  # Residual Layers

B = [3, 3, 3, 3, 3, 3]  # Residual blocks in Residual Layer i

C1 = 16

C = C1 * (2**np.arange(0, N, 1))  # channels in Residual Layer i

F = [5, 5, 5, 5, 5, 5]  # Conv. kernel size in Residual Layer i

# K Skip connection kernel size in Residual Layer i
# P Average pool kernel size
K, P = NBF2KP(N, B, F)