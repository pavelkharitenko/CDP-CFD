import numpy as np
import torch, sys
from scipy.spatial.transform import Rotation as R
sys.path.append('../../utils/')
from utils import *

class AgileEquivariantDataset(torch.utils.data.Dataset):
    def __init__(self, experiment_paths):
        self.exp_paths = experiment_paths
        self.extract_ndp_labels()

    def __len__(self):
        return self.N
    
    def __getitem__(self,index):
        return self.x[index,:], self.y[index,:]
        
    def extract_ndp_labels(self):
        x = np.empty((0,14))
        y = np.array([])
        for exp_path in self.exp_paths:
            self.N = 0
            
            
            data = np.load(exp_path)
            uav_1_states, uav_2_states = data['uav_1_states'], data['uav_2_states']
            #print(self.equivariant_lean_agile_transform(uav_1_states, uav_2_states).shape)
            x_i = equivariant_agile_transform(uav_1_states, uav_2_states)
            y_i = data['dw_forces'][:,2]
            x = np.vstack((x, x_i))
            y = np.append(y,y_i)

        print("extracted total X data of length:", len(x))
        print("extracted total Y labels of length:", len(y))

        self.x =  torch.tensor(x).to(torch.float32)
        self.y =  torch.tensor(y).to(torch.float32)
        self.N = len(self.x)

class AgileVelDataset(torch.utils.data.Dataset):
    def __init__(self, experiment_paths):
        self.exp_paths = experiment_paths
        self.extract_ndp_labels()

    def __len__(self):
        return self.N
    
    def __getitem__(self,index):
        return self.x[index,:], self.y[index,:]
        
    def extract_ndp_labels(self):
        x = np.empty((0,8))
        y = np.array([])
        for exp_path in self.exp_paths:
            self.N = 0
            
            
            data = np.load(exp_path)
            uav_1_states, uav_2_states = data['uav_1_states'], data['uav_2_states']
            #print(self.equivariant_lean_agile_transform(uav_1_states, uav_2_states).shape)
            equiv_transform = equivariant_agile_transform(uav_1_states, uav_2_states)
            x_i = equiv_transform[:,[0,1,8,9,10,11,12,13]]
            y_i = data['dw_forces'][:,2]
            x = np.vstack((x, x_i))
            y = np.append(y,y_i)

        print("extracted total X data of length:", len(x))
        print("extracted total Y labels of length:", len(y))

        self.x =  torch.tensor(x).to(torch.float32)
        self.y =  torch.tensor(y).to(torch.float32)
        self.N = len(self.x)


class AgileContinousDataset(torch.utils.data.Dataset):
    def __init__(self, experiment_paths):
        self.exp_paths = experiment_paths
        self.extract_ndp_labels()

    def __len__(self):
        return self.N
    
    def __getitem__(self,index):
        return self.x[index,:], self.y[index,:]
        
    def extract_ndp_labels(self):
        x = np.empty((0,14))
        y = np.empty((0,3))
        for exp_path in self.exp_paths:
            self.N = 0
            
            
            data = np.load(exp_path)
            uav_1_states, uav_2_states = data['uav_1_states'], data['uav_2_states']
            
            dw_x, dw_y, dw_z = extract_dw_forces(uav_2_states)

            x_dw_ct = continous_transform(equivariant_agile_transform(uav_1_states, uav_2_states))
            x_i = x_dw_ct[:,:14]
            dw_xy = x_dw_ct[:,14:16] # use transformed xy

            #print("shape dw_xy:", dw_xy.shape)
            #print("shape dw_z:", dw_z.shape)

            y_i = np.column_stack((dw_xy, dw_z))
            #print("shape y_i:", y_i.shape)
            x = np.vstack((x,x_i))
            y = np.vstack((y,y_i))

        print("extracted total X data of length:", len(x))
        print("extracted total Y labels of length:", len(y))
        self.x =  torch.tensor(x).to(torch.float32)
        self.y =  torch.tensor(y).to(torch.float32)
        #self.y =  torch.tensor(y).to(torch.float32).unsqueeze(1)
        self.N = len(self.x)

