import torch
import torch.nn as nn


# input format: x = [pos_rel, vel_rel] (relative state vector)
# output format: y = f_dw_pred xyz-force of downwash


class DWPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(6,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,128),
            nn.ReLU(),
            nn.Linear(128,3)
        )

    def forward(self,x):
        #x = self.flatten(x)
        return self.linear_relu_stack(x)
    
#model = DWPredictor().to("cuda")
#print(model)