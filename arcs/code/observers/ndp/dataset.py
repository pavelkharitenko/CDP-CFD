import numpy as np
import torch, sys
sys.path.append('../../utils/')
from utils import *

class DWDataset(torch.utils.data.Dataset):
    def __init__(self, experiment_paths, shuffle=False):
        self.exp_paths = experiment_paths
        self.extract_ndp_labels()

    def __len__(self):
        return self.N
    
    def __getitem__(self,index):
        return self.x[index,:], self.y[index,:]
        
    def extract_ndp_labels(self):
        self.N = 0
        x, y = [], []
        for exp_path in self.exp_paths:
            exp = load_forces_from_dataset(exp_path)
            uav_1, uav_2 = exp['uav_list']
            x_i, y_i = extract_labeled_dataset_ndp([uav_1, uav_2], exp['bias'])
            x.extend(x_i)
            y.extend(y_i)
            print("######################")

            print("######### x sample:", x[:5])
            print("######### y sample:", y[:5])

            print("extracted new data of lenghts:", len(x))
            print("extracted new data of lenghts:", len(y))

            
            
    
        self.x =  torch.tensor(x).to(torch.float32)
        self.y =  torch.tensor(y).to(torch.float32)

        self.N = len(self.x)

        #print("### data samples", self.x[:5,:])
        #print("### data samples", self.y[:5,:])


        






