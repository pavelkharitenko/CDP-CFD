import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class FeedForwardNN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.layer1 = nn.Linear(in_dim, 64)
        self.layer2 = nn.Linear(64,64)
        self.layer3 = nn.Linear(64, out_dim)

    
    def forward(self, x):
        if isinstance(x, np.ndarray): # check if not tensor, convert
            x = torch.tensor(x, dtype=torch.float)

        activation1 = F.relu(self.layer1(x))
        activation2 = F.relu(self.layer2(activation1))
        output = self.layer3(self.layer2(activation2))

        return output

