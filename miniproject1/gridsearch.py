import numpy as np
import netconfig
import pickle

class gridSearch():
    def __init__(self):
        return

    def getGridConfigs(self):
        configs = []
        #change N
        C1 = 16
        for N in range(2, 6):
            C = C1 * (2 ** np.arange(0, N, 1))
            B = 2 * np.ones(N).astype(int)
            F = 3 * np.ones(N).astype(int)
            K, P = netconfig.NBF2KP(N, B, F)
            config = {'N': N,
                      'B': B,
                      'C1': C1,
                      'F': F,
                      'K': K,
                      'P': P
                      }
            configs.append(config)
            print(config)

        #change C1
        N=4
        B=2*np.ones(N).astype(int)
        F = 3 * np.ones(N).astype(int)
        K, P = netconfig.NBF2KP(N, B, F)
        for C1 in [8,16,32,64,128]:
            config = {'N': N,
                      'B': B,
                      'C1': C1,
                      'F': F,
                      'K': K,
                      'P': P
                      }
            configs.append(config)
            print(config)

        #change B
        N = 4
        C1 = 64
        F = 3 * np.ones(N).astype(int)
        for b in [1,2,3,4,5]:
            B = b*np.ones(N).astype(int)
            K, P = netconfig.NBF2KP(N, B, F)
            config = {'N': N,
                      'B': B,
                      'C1': C1,
                      'F': F,
                      'K': K,
                      'P': P
                      }
            configs.append(config)
            print(config)

        #change F
        N = 4
        C1 = 64
        B = 2 * np.ones(N).astype(int)
        for f in [1, 3, 5, 7]:
            F = f*np.ones(N).astype(int)
            K, P = netconfig.NBF2KP(N, B, F)
            config = {'N': N,
                      'B': B,
                      'C1': C1,
                      'F': F,
                      'K': K,
                      'P': P
                      }
            configs.append(config)
            print(config)

        print("configs cnt", len(configs))

        with open('configs', 'wb') as f:
            pickle.dump(configs, f)

gs = gridSearch()
gs.getGridConfigs()

