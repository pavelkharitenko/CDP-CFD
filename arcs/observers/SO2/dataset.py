import numpy as np
import torch, sys
sys.path.append('../../utils/')
from utils import *
from model import *

class SO2Dataset(torch.utils.data.Dataset):
    def __init__(self, experiment_path):
        self.exp_paths = experiment_path
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
        x = np.empty((0,9))
        y = np.empty((0,3))
        for exp_path in self.exp_paths:
            self.N = 0
            
            data = np.load(exp_path)
            uav_1_states, uav_2_states = data['uav_1_states'], data['uav_2_states']
            
            delta_p_list = uav_1_states[:,:3] - uav_2_states[:,:3] 
            v_A_list = uav_1_states[:,3:6]
            v_B_list = uav_2_states[:,3:6]

            # compute network input via feature mapping function h
            h_features = h_mapping(delta_p_list, v_B_list, v_A_list)
            # include dp for F (to recover absolute angle of downwash)
            x_i = np.column_stack((h_features, delta_p_list))

            dw_x,dw_y,dw_z = extract_dw_forces(uav_2_states)
            y_i = np.column_stack((dw_x, dw_y, dw_z))

            x = np.vstack((x, x_i))
            y = np.vstack((y,y_i))


        #print("extracted and mapped total X data with length:", len(x))
        #print("extracted total Y labels with length:", len(y))  
    
        self.x =  torch.tensor(x).to(torch.float32)
        self.y =  torch.tensor(y).to(torch.float32)

        self.N = len(self.x)






