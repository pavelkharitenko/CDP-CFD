import numpy as np
import torch

class DWDataset(torch.utils.data.Dataset):
    def __init__(self, N):
        
        self.N = N
        x = np.empty((self.N,6))
        x[:] = np.array([0,0,0,0,0,0])

        y = np.empty((self.N,3))
        y[:] = np.array([0,8,6.5])

        self.x =  torch.tensor(x).to(torch.float32)
        self.y =  torch.tensor(y).to(torch.float32)

    def __len__(self):
        return self.N
    
    def __getitem__(self,index):
        return self.x[index,:], self.y[index,:]
        






