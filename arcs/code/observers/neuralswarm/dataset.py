import numpy as np
import torch, sys
sys.path.append('../../utils/')
from utils import *

class NSDataset(torch.utils.data.Dataset):
    def __init__(self, experiment_paths, from_both_uavs=False):
        self.exp_paths = experiment_paths
        if from_both_uavs:
            self.extract_ns_labels_twice()
        else:
            self.extract_ns_labels()

    def __len__(self):
        return self.N
    
    def __getitem__(self,index):
        return self.x[index,:], self.y[index,:]
        
    def extract_ns_labels(self):
        self.N = 0
        x, y = [], []
        for exp_path in self.exp_paths:
            exp = load_forces_from_dataset(exp_path)
            uav_1, uav_2 = exp['uav_list']
            if "bias" in exp:
                x_i, y_i = extract_labeled_dataset_ns([uav_1, uav_2], exp['bias'])
            else:
                x_i, y_i = extract_labeled_dataset_ns([uav_1, uav_2])

            #print("### Max value rec.:", np.max([np.abs(y_i_j[2]) for y_i_j in y_i]))
            #print("### Max value rec.:", y_i[-4000:-3990])

            x.extend(x_i)
            y.extend(y_i)
            #print("######################")

            

        #print("extracted total X data of length:", len(x))
        #print("extracted total Y labels of length:", len(y))

            
            
    
        self.x =  torch.tensor(x).to(torch.float32)
        self.y =  torch.tensor(y).to(torch.float32)

        self.N = len(self.x)

        #print("### data samples", self.x[:5,:])
        #print("### data samples", self.y[:5,:])


    def extract_ns_labels_twice(self):
        self.N = 0
        x, y = [], []
        for exp_path in self.exp_paths:
            exp = load_forces_from_dataset(exp_path)
            uav_1, uav_2 = exp['uav_list']
            if "bias" in exp:
                x_i, y_i = extract_labeled_dataset_ns([uav_1, uav_2], exp['bias'])
            else:
                x_i, y_i = extract_labeled_dataset_ns([uav_1, uav_2])

            #print("### Max value rec.:", np.max([np.abs(y_i_j[2]) for y_i_j in y_i]))
            #print("### Max value rec.:", y_i[-4000:-3990])

            x.extend(x_i)
            y.extend(y_i)

            
            if "bias" in exp:
                x_i, y_i = extract_labeled_dataset_ns([uav_2, uav_1], exp['bias'])
            else:
                x_i, y_i = extract_labeled_dataset_ns([uav_2, uav_1])

            x.extend(x_i)
            y.extend(y_i)
            #print("######################")

            #print("######### x sample:", x[:5])
            #print("######### y sample:", y[:5])

        #print("extracted total data of lenghts:", len(x))
        print("extracted total data of lenghts:", len(y))

        #print("### Max value rec.:", y_i[300:350])
            
            
    
        self.x =  torch.tensor(np.array(x)).to(torch.float32)
        self.y =  torch.tensor(np.array(y)).to(torch.float32)

        self.N = len(self.x)






