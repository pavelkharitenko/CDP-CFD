
import torch
import torch.nn as nn
import torch.nn.functional as F

def init_layer_uniform(layer: nn.Linear, init_w: float = 3e-3) -> nn.Linear:
    layer.weight.data.uniform(-init_w, init_w)
    layer.bias.data.uniform(-init_w, init_w)
    return layer


class Actor(nn.Module):

    def __init__(self, in_dim, out_dim, hidden_dim = 256):
        

        self.hidden1 = nn.Linear(in_dim, hidden_dim)
        self.hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.log_std_layer = init_layer_uniform(nn.Linear(hidden_dim, out_dim))
        self.mean_layer = init_layer_uniform(nn.Linear(hidden_dim, out_dim))

        self.log_std_min = -20
        self.log_std_max = 2

    def forward(self, state: torch.Tensor):
        x = F.relu(self.hidden1(state))
        x = F.relu(self.hidden2(x))

        # compute mean and log std

        mean = self.mean_layer(x).tanh()
        log_std = self.log_std_layer(x).tanh()

        # soft clamp log std:
        log_std = self.log_std_min + 0.5* (self.log_std_max - self.log_std_min) * (log_std + 1)


        # sample actions
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mean, std)
        z = dist.rsample()

        action = z.tanh()
        log_prob = dist.log_prob(z) - torch.log(1 - action.pow(2) + 1e-7)
        log_prob = log_prob.sum(-1, keepdim=True)

        return action, log_prob
    
class CriticQ(nn.Module):
    def __init__(self, in_dim, hidden_dim = 256):
        super(CriticQ, self).__init__()

        self.hidden1 = nn.Linear(in_dim, hidden_dim)
        self.hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = init_layer_uniform(nn.Linear(hidden_dim, 1))

    def forward(self, state, action):

        x = torch.cat((state, action), dim=1)
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        return self.out(x)


class CriticV(nn.Module):
    def __init__(self, in_dim, hidden_dim = 256):
        super(CriticV, self).__init__()
        self.hidden1 = nn.Linear(in_dim, hidden_dim)
        self.hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = init_layer_uniform(nn.Linear(hidden_dim, 1))

    def forward(self, state):
        x = F.relu(self.hidden1(state))
        x = F.relu(self.hidden2(x))
        return self.out(x)

    