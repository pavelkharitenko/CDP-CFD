#----------------------------------
# DW predictor based on Smith et al. SO(2)-Equivariant Downwash Models for Close Proximity Flight
#----------------------------------
import torch
import torch.nn as nn



class ShallowEquivariantPredictor(nn.Module):
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

def h(rel_pos, v_suff, v_prod):
    pass
