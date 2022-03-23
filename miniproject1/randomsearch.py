import numpy as np
import random
import netconfig
import pickle

class randomSearch():
    def __init__(self):
        return

    def getRandomConfigs(self, n):
        valid_para_cnt = 0

        configs = []
        for i in range(0, n):
            N = random.randint(3, 6)
            C1 = random.choice([8, 16, 32, 64])
            C = C1 * (2**np.arange(0, N, 1))
            B = []
            F = []
            for j in range(0, N):
                b = random.randint(2,3)
                B.append(b)
                f = random.choice([3, 5])
                F.append(f)
            K, P = netconfig.NBF2KP(N, B, F)
            if netconfig.verifyPara(N, C, B, F, K):
                config = {'N':N,
                          'B':B,
                          'C1':C1,
                          'F':F,
                          'K':K,
                          'P':P
                          }
                if config not in configs:
                    valid_para_cnt += 1
                    configs.append(config)
                    print(config)
        with open('configs', 'wb') as f:
            pickle.dump(configs, f)

        print("valid para count", valid_para_cnt)
        return

#rs = randomSearch()
#rs.getRandomConfigs(1000)