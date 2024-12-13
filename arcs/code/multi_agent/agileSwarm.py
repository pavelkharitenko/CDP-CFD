

# GNN procedure:

# design node2edge update, design edge2node update
# node update should have aggr. function, may also be a sum(rho()) where rho is an internal network
# design 4 dimesions: node dim, edge dim, and node mlp dim (could be node_dim + edge dim, or in addition pass node_0)
# for edge_mlp dim: could be node_dim * 2, edge_dim (old edge state)

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import MessagePassing
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------
# GNN Model Definition
# ---------------------------
class UAVGNN(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, output_dim, num_layers):
        super(UAVGNN, self).__init__()
        self.num_layers = num_layers
        
        # Edge-to-Node MLP (initialization)
        self.edge_to_node_mlp = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Message Passing Layers
        self.message_passing_layers = nn.ModuleList([
            MessagePassingLayer(node_dim=hidden_dim, edge_dim=hidden_dim) for _ in range(num_layers)
        ])
        
        # Final Prediction MLP
        self.output_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x, edge_index, edge_attr):
        """
        x: node features [num_nodes, node_dim]
        edge_index: connectivity [2, num_edges]
        edge_attr: edge features [num_edges, edge_dim]
        """
        # Step 1: Edge-to-Node initialization
        edge_messages = self.edge_to_node_mlp(edge_attr)  # [num_edges, hidden_dim]
        node_features = torch.zeros((x.size(0), edge_messages.size(1)), device=x.device)
        
        # Aggregate messages to initialize node features
        for i, (src, dst) in enumerate(edge_index.t()):
            node_features[dst] += edge_messages[i]
        
        # Step 2: Message Passing
        for layer in self.message_passing_layers:
            node_features = layer(node_features, edge_index, edge_attr)
        
        # Step 3: Predict external forces
        predictions = self.output_mlp(node_features)  # [num_nodes, output_dim]
        return predictions

class MessagePassingLayer(MessagePassing):
    def __init__(self, node_dim, edge_dim):
        super(MessagePassingLayer, self).__init__(aggr='mean')
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * node_dim + edge_dim, node_dim),
            nn.ReLU(),
            nn.Linear(node_dim, node_dim)
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(node_dim, node_dim),
            nn.ReLU()
        )
        
    def forward(self, x, edge_index, edge_attr):
        """
        x: node features [num_nodes, node_dim]
        edge_index: connectivity [2, num_edges]
        edge_attr: edge features [num_edges, edge_dim]
        """
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        """
        x_i: node features of target nodes
        x_j: node features of source nodes
        edge_attr: edge features
        """
        # Concatenate node features and edge attributes
        input_features = torch.cat([x_i, x_j, edge_attr], dim=-1)
        return self.edge_mlp(input_features)

    def update(self, aggr_out):
        return self.node_mlp(aggr_out)

# ---------------------------
# Data Preparation
# ---------------------------
def create_uav_graph_data(states, labels):
    """
    Convert absolute state vectors to graph data.
    states: [n_samples, n_uavs, 14]
    labels: [n_samples, n_uavs, 1]
    """
    data_list = []
    n_samples, n_uavs, _ = states.shape
    
    for i in range(n_samples):
        # Node features: leave empty for edge initialization
        node_features = torch.zeros((n_uavs, 1))
        
        # Edge features: relative position and velocity
        edge_index = []
        edge_attr = []
        for u in range(n_uavs):
            for v in range(n_uavs):
                if u != v:
                    edge_index.append([u, v])
                    rel_pos = states[i, v, :3] - states[i, u, :3]  # Relative position
                    rel_vel = states[i, v, 3:6] - states[i, u, 3:6]  # Relative velocity
                    edge_attr.append(torch.cat([rel_pos, rel_vel], dim=-1))
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.stack(edge_attr)
        
        # Ground truth labels for nodes
        y = torch.tensor(labels[i], dtype=torch.float)
        
        # Graph data
        data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr, y=y)
        data_list.append(data)
    return data_list

# ---------------------------
# Training Loop
# ---------------------------
def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data.x, data.edge_index, data.edge_attr)
        loss = criterion(output, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

# ---------------------------
# Main Execution
# ---------------------------
if __name__ == "__main__":
    # Simulated data
    n_samples = 100
    n_uavs = 5
    state_dim = 14
    states = torch.randn((n_samples, n_uavs, state_dim))  # Absolute positions, velocities
    labels = torch.randn((n_samples, n_uavs, 1))  # Ground truth external forces

    # Prepare graph data
    data_list = create_uav_graph_data(states, labels)
    loader = DataLoader(data_list, batch_size=4, shuffle=True)

    # Model and Training Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UAVGNN(node_dim=1, edge_dim=6, hidden_dim=32, output_dim=1, num_layers=3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # Training Loop
    epochs = 50
    for epoch in range(epochs):
        loss = train(model, loader, optimizer, criterion, device)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss:.4f}")
    
    # Inference and Plotting
    model.eval()
    sample = data_list[0].to(device)
    with torch.no_grad():
        predictions = model(sample.x, sample.edge_index, sample.edge_attr)
        predictions = predictions.cpu().numpy()
        ground_truth = sample.y.cpu().numpy()
    
    # Plot ground truth vs predictions
    plt.figure(figsize=(8, 6))
    plt.plot(range(n_uavs), ground_truth, 'bo-', label='Ground Truth')
    plt.plot(range(n_uavs), predictions, 'ro-', label='Predictions')
    plt.xlabel('UAV Index')
    plt.ylabel('External Force')
    plt.legend()
    plt.title('Predicted vs Ground Truth External Forces')
    plt.show()
