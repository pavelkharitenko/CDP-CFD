import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import random
from dataset import *
from scipy.signal import savgol_filter


def extract_data_sindy(dataset, normalize=True):
    # Assume dataset.x_arr and dataset.y_arr are already loaded
    states = dataset.x_arr  # Shape: (n_train, 14)
    forces = dataset.y_arr

    

    #states = dataset.x_arr[:,[0,1,4,5,6,7,8,9,10,11,12,13]]


    #if normalize:
    #    scaler = StandardScaler()
    #    states = scaler.fit_transform(states)  # Scale the input data
    #    forces = scaler.fit_transform(forces.reshape(-1,1))  # Scale the input data

    

    threshold = 0.1
    #states, forces, timestamp_lists, split_times = split_sequences_by_threshold(states, forces, threshold)




    # create timestamp array for each trajectory
    timestamps = []
    for state_array in states:
        time = 0.0
        n_states = state_array.shape[0]  # Get the number of states (time steps)
        time_array = np.zeros(n_states)
        for i in range(n_states):
            time_array[i] = time
            time += 0.005
        timestamps.append(time_array)


    t = timestamps



    return states, forces, t





class NNARXDataset(Dataset):
    def __init__(self, states, forces, lags=5):
        """
        states: NumPy array of shape (n_samples, n_states), e.g., (n, 14)
        forces: NumPy array of shape (n_samples, n_forces), e.g., (n, 3)
        lags: Number of previous timesteps to use as inputs
        """
        self.lags = lags
        
        # Get the total number of time steps (n_samples) and feature dimensions (n_states, n_forces)
        n_samples, n_states = states.shape
        _, n_forces = forces.shape
        
        # Create lagged states and forces
        # Create the lagged states: (n_samples - lags, lags * n_states)
        lagged_states = np.array([states[i - self.lags:i].flatten() for i in range(self.lags, n_samples)])
        
        # Create the lagged forces: (n_samples - lags, lags * n_forces)
        lagged_forces = np.array([forces[i - self.lags:i].flatten() for i in range(self.lags, n_samples)])
        
        # Stack lagged states (n_samples - lags, (lags * n_states) + (lags * n_forces))
        inputs = np.hstack([lagged_states])

        # Create the outputs: forces at time step i, shape: (n_samples - lags, n_forces)
        #outputs = forces[self.lags:]
        
        # Convert inputs and outputs to torch tensors
        self.inputs = torch.tensor(inputs, dtype=torch.float32)
        outputs = forces[self.lags:, 2].reshape(-1, 1)  # Reduce output dimension from (n, 3) to (n, 1)
        self.outputs = torch.tensor(outputs, dtype=torch.float32)


    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]

# --- Step 2: Neural Network Architecture ---
class NNARXModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        """
        input_dim: Number of input features (lags * state_dim + lags * force_dim)
        hidden_dim: Number of hidden units
        """
        super(NNARXModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Single output: y(t+1)
        )

    def forward(self, x):
        return self.network(x)


# --- Step 3: Data Preparation ---
def prepare_data(dataset, lags=5, normalize=True):
    states, forces, t = extract_data_sindy(dataset, normalize)
    train_dataset = NNARXDataset(states, forces, lags=lags)
    return train_dataset


# --- Step 4: Training Function ---
def train_nn(model, train_loader, optimizer, criterion, epochs=20):
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.6f}")


# --- Step 5: Evaluation Function ---
def evaluate_nn(model, eval_loader):
    model.eval()
    predictions = []
    actuals = []

    with torch.no_grad():
        for inputs, targets in eval_loader:
            outputs = model(inputs)
            predictions.append(outputs.numpy())
            actuals.append(targets.numpy())

    predictions = np.vstack(predictions)
    actuals = np.vstack(actuals)

    mse = mean_squared_error(actuals, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(actuals, predictions)

    print(f"Evaluation Results: RMSE: {rmse:.4f}, R²: {r2:.4f}")
    return actuals, predictions


# --- Step 6: Main Script ---
if __name__ == "__main__":
    # Set random seed
    random.seed(2)
    torch.manual_seed(2)
    np.random.seed(2)

    dataset = AgileContinousDataset([
        r"C:\Users\admin\Desktop\IDP\CDP-CFD\arcs\code\data_collection\agile_manuevers\1_flybelow\speeds_05_20\raw_data_1_flybelow_200Hz_80_005_len68899ts_103_iterations.npz",
        r"C:\Users\admin\Desktop\IDP\CDP-CFD\arcs\code\data_collection\agile_manuevers\2_flyabove\speeds_05_20\raw_data_2_flyabove_200Hz_80_005_len7636ts_12_iterations.npz",
        r"C:\Users\admin\Desktop\IDP\CDP-CFD\arcs\code\data_collection\agile_manuevers\2_flyabove\speeds_05_20\raw_data_2_flyabove_200Hz_80_005_len65911ts_91_iterations.npz",
        #r"C:\Users\admin\Desktop\IDP\CDP-CFD\arcs\code\data_collection\agile_manuevers\3_swapping\speeds_05_20\raw_data_3_swapping_200Hz_80_005_len60694ts_100_iterations.npz",
        #r"C:\Users\admin\Desktop\IDP\CDP-CFD\arcs\code\data_collection\agile_manuevers\3_swapping\speeds_20_40\raw_data_3_swapping_fast_200Hz_80_005_len56629ts_173_iterations.npz",
        ])
    
    dataset_eval = AgileContinousDataset([
        r"C:\Users\admin\Desktop\IDP\CDP-CFD\arcs\code\data_collection\agile_manuevers\1_flybelow\speeds_05_20\raw_data_1_flybelow_200Hz_80_005_len34951ts_51_iterations_testset.npz",
        r"C:\Users\admin\Desktop\IDP\CDP-CFD\arcs\code\data_collection\agile_manuevers\2_flyabove\speeds_05_20\raw_data_2_flyabove_200Hz_80_005_len32956ts_46_iterations_testset.npz",
        r"C:\Users\admin\Desktop\IDP\CDP-CFD\arcs\code\data_collection\agile_manuevers\3_swapping\speeds_05_20\raw_data_3_swapping_200Hz_80_005_len29736ts_52_iterations_testset.npz",
        r"C:\Users\admin\Desktop\IDP\CDP-CFD\arcs\code\data_collection\agile_manuevers\3_swapping\speeds_20_40\raw_data_3_swapping_fast_200Hz_80_005_len20802ts_67_iterations_testset.npz",
        ])

    # Data preparation
    lags = 1  # Number of timesteps to look back
    train_dataset = prepare_data(dataset, lags=lags)
    eval_dataset = prepare_data(dataset_eval, lags=lags)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=64, shuffle=False)

    # Model initialization
    input_dim = train_dataset.inputs.shape[1]
    print("Input shape is:", input_dim)
    model = NNARXModel(input_dim=input_dim, hidden_dim=64)

    # Optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    # Train the model
    print("Training NNARX model...")
    train_nn(model, train_loader, optimizer, criterion, epochs=20)

    # Evaluate the model
    print("Evaluating NNARX model...")
    actuals, predictions = evaluate_nn(model, eval_loader)

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(actuals, label="True Forces", linewidth=2)
    plt.plot(predictions, label="Predicted Forces (NNARX)", linewidth=2, linestyle="--")
    plt.xlabel("Time Step")
    plt.ylabel("Force")
    plt.title("True vs Predicted Disturbance Forces")
    plt.legend()
    plt.show()
