import numpy as np
import torch

class DWDataset(torch.utils.data.Dataset):
    def __init__(self, N):
        
        self.N = N
        x = np.empty((N,6))
        x[:int(self.N/2.0)] = np.array([0,0,1,0,0,0])
        x[int(self.N/2.0):] = np.array([0,0,-1,0,0,0])

        y = np.empty((self.N,3))
        y[:] = np.array([5,0,0])

        y[:int(self.N/2.0)] = np.array([0,0,-3])
        y[int(self.N/2.0):] = np.array([0,0, 0.8])

        self.x =  torch.tensor(x).to(torch.float32)
        self.y =  torch.tensor(y).to(torch.float32)

    def __len__(self):
        return self.N
    
    def __getitem__(self,index):
        return self.x[index,:], self.y[index,:]
        






