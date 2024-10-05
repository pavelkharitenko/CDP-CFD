import numpy as np
import torch
from utils import *

class DWDataset(torch.utils.data.Dataset):
    def __init__(self, experiment_path, shuffle=False):
        
        self.exp_path = experiment_path
        self.extract_ndp_labels(shuffle)



        #self.N = N

        #x = np.empty((N,6))
        #x[:int(self.N/2.0)] = np.array([0,0,1,0,0,0])
        #x[int(self.N/2.0):] = np.array([0,0,-1,0,0,0])

        #y = np.empty((self.N,3))
        ##y[:] = np.array([5,0,0])

        #y[:int(self.N/2.0)] = np.array([0,0,-3])
        #y[int(self.N/2.0):] = np.array([0,0, 0.8])

        #self.x =  torch.tensor(x).to(torch.float32)
        #self.y =  torch.tensor(y).to(torch.float32)

    def __len__(self):
        return self.N
    
    def __getitem__(self,index):
        return self.x[index,:], self.y[index,:]
        
    def extract_ndp_labels(self, shuffle):
        exp = load_forces_from_dataset(self.exp_path)
        uav_1, uav_2 = exp['uav_list']
        x,y = extract_labeled_dataset_ndp([uav_1, uav_2])
        x_2, y_2 = extract_labeled_dataset_ndp([uav_2, uav_1])
        x = np.append(x, x_2, axis=0)
        y = np.append(y, y_2, axis=0)
        #print("###", xy_labeled_data)
        #xy_labeled_data  = np.array(xy_labeled_data)
        #if shuffle:
        #    np.random.shuffle(xy_labeled_data)
        
        #self.dataset = torch.tensor(xy_labeled_data).to(torch.float32)
        self.x =  torch.tensor(x).to(torch.float32)
        self.y =  torch.tensor(y).to(torch.float32)
        self.N = len(self.x)

        #print("### data samples", self.x[:5,:])
        #print("### data samples", self.y[:5,:])


        






