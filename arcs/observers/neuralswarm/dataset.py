import numpy as np
import torch, sys
sys.path.append('../../utils/')
from utils import *

class NeuralSwarmDataset(torch.utils.data.Dataset):
    def __init__(self, experiment_paths):
        self.exp_paths = experiment_paths
        self.extract_nrlsw_labels()

    def __len__(self):
        return self.N
    
    def __getitem__(self,index):
        return self.x[index,:], self.y[index,:]
        
    def extract_nrlsw_labels(self):
        x = np.empty((0,6))
        y = np.empty([])
        for exp_path in self.exp_paths:
            self.N = 0
            
            
            data = np.load(exp_path)
            uav_1_states, uav_2_states = data['uav_1_states'], data['uav_2_states']
            
            x_i = uav_1_states[:,:6] - uav_2_states[:,:6] # compute rel. state of pos. and vel.
            
            dw_x,dw_y,dw_z = extract_dw_forces(uav_2_states)
            

            x = np.vstack((x, x_i))
            y = np.append(y,dw_z)



        print("extracted total X data of length:", len(x))
        print("extracted total Y labels of length:", len(y))

            
            
    
        self.x =  torch.tensor(x).to(torch.float32)
        self.y =  torch.tensor(y).to(torch.float32).unsqueeze(1)

        self.N = len(self.x)

        #print("### data samples", self.x[:5,:])
        #print("### data samples", self.y[:5,:])




