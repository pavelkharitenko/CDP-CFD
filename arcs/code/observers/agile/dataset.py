import numpy as np
import torch, sys
sys.path.append('../../utils/')
from utils import *

class DWDataset(torch.utils.data.Dataset):
    def __init__(self, experiment_paths):
        self.exp_paths = experiment_paths
        self.extract_ndp_labels()

    def __len__(self):
        return self.N
    
    def __getitem__(self,index):
        return self.x[index,:], self.y[index,:]
        
    def extract_ndp_labels(self):
        self.N = 0
        x, y = [], []
        
        data = np.load(self.exp_paths)
        uav_1_states, uav_2_states = data['uav_1_states'], data['uav_2_states']
        
        x = uav_1_states[:,:6] - uav_2_states[:,:6]
        y = data['dw_forces']

        


        #print("extracted total X data of length:", len(x))
        #print("extracted total Y labels of length:", len(y))

            
            
    
        self.x =  torch.tensor(x).to(torch.float32)
        self.y =  torch.tensor(y).to(torch.float32)

        self.N = len(self.x)

        #print("### data samples", self.x[:5,:])
        #print("### data samples", self.y[:5,:])








