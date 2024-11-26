import numpy as np
import torch, sys
sys.path.append('../../utils/')
from utils import *
from model import *

class SO2Dataset(torch.utils.data.Dataset):
    def __init__(self, experiment_path):
        self.exp_path = experiment_path
        self.extract_so2_labels()
        

    def __len__(self):
        return self.N
    
    def __getitem__(self,index):
        return self.x[index,:], self.y[index,:]
        


    def extract_so2_labels(self):
        """
        Extract relative position (uav1 - uav2), uav2 vel, uav1 vel, and dw forces
        """
        self.N = 0
        x, y = [], []
        

        data = np.load(self.exp_path)
        uav_1_states, uav_2_states = data['uav_1_states'], data['uav_2_states']
        
        dx = uav_1_states[:,:3] - uav_2_states[:,:3]

        relpos_vel_b_vel_a = list(zip(dx, uav_2_states[:,3:6], uav_1_states[:,3:6]))
        x = [ h(dp_vb_va[0], dp_vb_va[1], dp_vb_va[2]) + [dp_vb_va[0][0], dp_vb_va[0][1], dp_vb_va[0][2]] for dp_vb_va in relpos_vel_b_vel_a]
        y = data['dw_forces']
            

        print("extracted total data of lenghts:", len(x))
        print("extracted total data of lenghts:", len(y))
    
        self.x =  torch.tensor(np.array(x)).to(torch.float32)
        self.y =  torch.tensor(np.array(y)).to(torch.float32)

        self.N = len(self.x)





