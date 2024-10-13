import numpy as np
import torch, sys
sys.path.append('../../utils/')
from utils import *
from model import *

class SO2Dataset(torch.utils.data.Dataset):
    def __init__(self, experiment_paths, extract_twice=False):
        self.exp_paths = experiment_paths
        if extract_twice:
            self.extract_so2_labels_twice()
        else:
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
        for exp_path in self.exp_paths:
            exp = load_forces_from_dataset(exp_path)
            uav_1, uav_2 = exp['uav_list']

            rel_pos, v_suff, v_prod, dw_forces = extract_labeled_dataset_so2([uav_1, uav_2])
            dp_vb_va_list = list(zip(rel_pos, v_suff, v_prod))
            x.extend([ h(dp_vb_va[0], dp_vb_va[1], dp_vb_va[2]) + [dp_vb_va[0][0], dp_vb_va[0][1], dp_vb_va[0][2]] for dp_vb_va in dp_vb_va_list])

            y.extend(dw_forces)

        print("extracted total data of lenghts:", len(x))
        print("extracted total data of lenghts:", len(y))

    
        self.x =  torch.tensor(np.array(x)).to(torch.float32)
        self.y =  torch.tensor(np.array(y)).to(torch.float32)

        #print(self.x[:2])
        #print(self.y[:2])


        self.N = len(self.x)




    def extract_so2_labels_twice(self):
        """
        Extract relative position (uav1 - uav2), uav2 vel, uav1 vel, and dw forces
        """
        self.N = 0
        x, y = [], []
        for exp_path in self.exp_paths:
            exp = load_forces_from_dataset(exp_path)
            uav_1, uav_2 = exp['uav_list']

            rel_pos, v_suff, v_prod, dw_forces = extract_labeled_dataset_so2([uav_1, uav_2])
            dp_vb_va_list = list(zip(rel_pos, v_suff, v_prod))

            x.extend([ h(dp_vb_va[0], dp_vb_va[1], dp_vb_va[2]) + [dp_vb_va[0][0], dp_vb_va[0][1], dp_vb_va[0][2]] for dp_vb_va in dp_vb_va_list])
            y.extend(dw_forces)

            rel_pos, v_suff, v_prod, dw_forces = extract_labeled_dataset_so2([uav_2, uav_1])
            dp_vb_va_list = list(zip(rel_pos, v_suff, v_prod))

            x.extend([ h(dp_vb_va[0], dp_vb_va[1], dp_vb_va[2]) + [dp_vb_va[0][0], dp_vb_va[0][1], dp_vb_va[0][2]] for dp_vb_va in dp_vb_va_list])
            y.extend(dw_forces)



        print("extracted total data of lenghts:", len(x))
        print("extracted total data of lenghts:", len(y))

    
        self.x =  torch.tensor(np.array(x)).to(torch.float32)
        self.y =  torch.tensor(np.array(y)).to(torch.float32)

        #print(self.x[:2])
        #print(self.y[:2])


        self.N = len(self.x)





