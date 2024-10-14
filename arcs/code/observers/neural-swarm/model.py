#----------------------------------
# Deepsets permutation invariant model implemented from Shi et al. Neural-Swarm: 
# Decentralized Close-Proximity Multirotor Control Using Learned Interactions
#----------------------------------
import torch
import torch.nn as nn


class NeuralSwarmPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.MSELoss(reduction="sum")
        self.phi = nn.Sequential(
            nn.Linear(6,25),
            nn.ReLU(),
            nn.Linear(25,40),
            nn.ReLU(),
            nn.Linear(40,40),
            nn.ReLU(),
            nn.Linear(40,40)
        )
        self.rho = nn.Sequential(
            nn.Linear(40,40),
            nn.ReLU(),
            nn.Linear(40,40),
            nn.ReLU(),
            nn.Linear(40,40),
            nn.ReLU(),
            nn.Linear(40,1)
        )

    def forward(self,x): # x is (batch_size, set_size, input_dim)
        x = self.phi(x)  # (batch_size, set_size, output_dim)
        x_sum = x.sum(dim=1) # now after sum (batch_size, output_dim)
        return self.rho(x_sum) # (batch_size, output_dim)



    
