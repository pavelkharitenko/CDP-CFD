import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
from dataset import *
from scipy.signal import savgol_filter

device = "cuda" if torch.cuda.is_available() else "cpu"


def extract_data_sindy(dataset, normalize=True):
    # Assume dataset.x_arr and dataset.y_arr are already loaded
    states = dataset.x_arr  # Shape: (n_train, 14)
    forces = dataset.y_arr


    return states, forces





class NNARXDataset(Dataset):
    def __init__(self, states, forces, lags=5):
        """
        states: NumPy array of shape (n_samples, n_states), e.g., (n, 14)
        forces: NumPy array of shape (n_samples, n_forces), e.g., (n, 3)
        lags: Number of previous timesteps to use as inputs
        """

        self.lags = lags
        n_samples, n_states = states.shape
        _, n_forces = forces.shape
        
        # Initialize lists to store lagged states and forces
        lagged_states = []
        lagged_forces = []
        
        # For each time step, create lagged states and forces
        for i in range(self.lags, n_samples):
            # Create lagged states (each feature has its own lag)
            state_lags = states[i - self.lags:i]  # Shape (lags, n_states)
            force_lags = forces[i - self.lags:i]  # Shape (lags, n_forces)
            
            # Flatten the lagged states and forces, but keep them separate for each feature
            lagged_states.append(state_lags.flatten())  # Flatten each feature's lagged values
            lagged_forces.append(force_lags.flatten())  # Flatten forces similarly
        
        # Stack the lagged states and forces into input features
        lagged_states = np.array(lagged_states)
        lagged_forces = np.array(lagged_forces)

        
        # Combine states and forces to create the final input (lags * n_states + lags * n_forces)
        inputs = np.hstack([lagged_states])
        

        # Create the outputs: forces at time step i (with only the last force)
        outputs = forces[self.lags:, 2].reshape(-1, 1)  # Reduce output dimension from (n, 3) to (n, 1)
        
        # Convert inputs and outputs to torch tensors
        self.inputs = torch.tensor(inputs, dtype=torch.float32)
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
    states, forces = extract_data_sindy(dataset, normalize)
    train_dataset = NNARXDataset(states, forces, lags=lags)
    return train_dataset


# --- Step 4: Training Function ---
# --- Updated Training Function ---
def train_nn_with_validation(model, train_loader, val_loader, optimizer, criterion, epochs=20):
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        epoch_loss = 0.0
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        

        print(f"Epoch [{epoch+1}/{epochs}], Training Loss: {total_loss/len(train_loader):.6f}")
        print(f"Epoch [{epoch+1}/{epochs}], Epoch Loss: {total_loss/len(train_loader):.6f}")
        # After each epoch, evaluate on validation set
        if epoch % 5 == 0:
            test_nn(model, val_loader)




# --- Step 5: Evaluation Function ---
def test_nn(model, eval_loader):
    model.eval()

    with torch.no_grad():
        total_loss = 0.0
        for inputs, targets in eval_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()

    
    print(f"Validation Loss: {total_loss/len(eval_loader):.6f}")
    print(f"Epoch Validation Loss: {total_loss:.6f}")


# --- Step 5: Evaluation Function ---
def evaluate_nn(model, eval_loader):
    model.eval()
    predictions = []
    actuals = []

    with torch.no_grad():
        for inputs, targets in eval_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
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

    SAVE_MODEL = True

    lags = 15
    n_epochs = 20
    exp_name = init_experiment(f"NNARX-lags-{str(lags)}-123S")

    random.seed(2)
    torch.manual_seed(2)
    np.random.seed(2)

    dataset = AgileContinousDataset([
        r"C:\Users\admin\Desktop\IDP\CDP-CFD\arcs\code\data_collection\agile_manuevers\1_flybelow\speeds_05_20\raw_data_1_flybelow_200Hz_80_005_len68899ts_103_iterations.npz",
        r"C:\Users\admin\Desktop\IDP\CDP-CFD\arcs\code\data_collection\agile_manuevers\2_flyabove\speeds_05_20\raw_data_2_flyabove_200Hz_80_005_len7636ts_12_iterations.npz",
        r"C:\Users\admin\Desktop\IDP\CDP-CFD\arcs\code\data_collection\agile_manuevers\2_flyabove\speeds_05_20\raw_data_2_flyabove_200Hz_80_005_len65911ts_91_iterations.npz",
        r"C:\Users\admin\Desktop\IDP\CDP-CFD\arcs\code\data_collection\agile_manuevers\3_swapping\speeds_05_20\raw_data_3_swapping_200Hz_80_005_len60694ts_100_iterations.npz",
        r"C:\Users\admin\Desktop\IDP\CDP-CFD\arcs\code\data_collection\agile_manuevers\3_swapping\speeds_20_40\raw_data_3_swapping_fast_200Hz_80_005_len56629ts_173_iterations.npz",
        ])
    
    dataset_eval = AgileContinousDataset([
        r"C:\Users\admin\Desktop\IDP\CDP-CFD\arcs\code\data_collection\agile_manuevers\1_flybelow\speeds_05_20\raw_data_1_flybelow_200Hz_80_005_len34951ts_51_iterations_testset.npz",
        r"C:\Users\admin\Desktop\IDP\CDP-CFD\arcs\code\data_collection\agile_manuevers\2_flyabove\speeds_05_20\raw_data_2_flyabove_200Hz_80_005_len32956ts_46_iterations_testset.npz",
        r"C:\Users\admin\Desktop\IDP\CDP-CFD\arcs\code\data_collection\agile_manuevers\3_swapping\speeds_05_20\raw_data_3_swapping_200Hz_80_005_len29736ts_52_iterations_testset.npz",
        r"C:\Users\admin\Desktop\IDP\CDP-CFD\arcs\code\data_collection\agile_manuevers\3_swapping\speeds_20_40\raw_data_3_swapping_fast_200Hz_80_005_len20802ts_67_iterations_testset.npz",
        ])

    # Data preparation
    # Number of timesteps to look back
    train_dataset = prepare_data(dataset, lags=lags)
    eval_dataset = prepare_data(dataset_eval, lags=lags)



    train_data, val_data = train_test_split(train_dataset, test_size=0.25, shuffle=True)

    batch_size = 64



    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

    # Model initialization
    input_dim = train_dataset.inputs.shape[1]
    print("Input shape is:", input_dim)
    model = NNARXModel(input_dim=input_dim, hidden_dim=64)
    model = model.to(device)

    # Optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    # Train the model
    print("Training NNARX model...")
    train_nn_with_validation(model, train_loader, val_loader, optimizer, criterion, epochs=n_epochs)


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

    final_model_name = exp_name + str(n_epochs) + "_eps"  + ".pth"
    if SAVE_MODEL:
        print("Training finished, saving...", final_model_name)
        torch.save(model.state_dict(), final_model_name)
