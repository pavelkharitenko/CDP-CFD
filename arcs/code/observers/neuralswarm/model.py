#----------------------------------
# Deepsets permutation invariant model implemented from Shi et al. Neural-Swarm: 
# Decentralized Close-Proximity Multirotor Control Using Learned Interactions
#----------------------------------
import torch
import torch.nn as nn
import numpy as np

class NeuralSwarmPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        #self.loss_fn = nn.MSELoss(reduction="sum")
        self.loss_fn = nn.MSELoss()
        
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
        #print("input shape:", x.shape)
        x = self.phi(x)  # (batch_size, set_size, output_dim)
        #print("input shape after phi:", x.shape)

        #x_sum = x.sum(dim=1) # now after sum (batch_size, output_dim)
        #print("input shape after summing phi:", x.shape)
        # rho_x = self.rho(x_sum)
        return  self.rho(x) # (batch_size, output_dim)



    def evaluate(self, rel_state_vectors):
        """
        From R:6 (rel_pos, rel_vel) to R:3 (0, 0, dw_z) 
        """
        with torch.no_grad():
            inputs = torch.tensor(rel_state_vectors).to(torch.float32)
            dw_forces = self.forward(inputs).detach().cpu().numpy()
            
            padding = np.zeros((len(dw_forces), 3))
            padding[:,2] = dw_forces.squeeze()
            return padding


