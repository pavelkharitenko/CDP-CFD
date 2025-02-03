import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import random
from dataset import *
from model import DWPredictor

sys.path.append('../../utils/')
from utils import *
device = "cuda" if torch.cuda.is_available() else "cpu"



# --- Step 4: Training Function ---
def train_nn(model, train_loader, val_loader, optimizer, criterion, epochs=20):
    tr_losses = []
    val_losses = []
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.6f}")
        
        tr_losses.append(total_loss/len(train_loader))
        val_losses.append(test_nn(model, val_loader))
    return tr_losses, val_losses

# --- Step 5: Evaluation Function ---
def test_nn(model, eval_loader):
    model.eval()

    with torch.no_grad():
        total_loss = 0.0
        for inputs, targets in eval_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()

    
    print(f"Validation Loss: {total_loss/len(eval_loader):.6f}")
    return total_loss/len(eval_loader)
    #print(f"Epoch Validation Loss: {total_loss:.6f}")


# --- Step 5: Evaluation Function ---
def evaluate_nn(model, eval_loader):

    model.eval()
    model.to("cpu")
    predictions = []
    actuals = []

    with torch.no_grad():
        for inputs, targets in eval_loader:
            outputs = model(inputs)
            #print(outputs.numpy().shape)
            predictions.append(outputs.numpy())
            actuals.append(targets.numpy())
            #print(len(predictions))

    predictions = np.vstack(predictions)[:,2]
    actuals = np.vstack(actuals)[:,2]

    mse = mean_squared_error(actuals, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(actuals, predictions)

    print(f"Evaluation Results: RMSE: {rmse:.4f}, R²: {r2:.4f}")
    return actuals, predictions


# --- Step 6: Main Script ---
if __name__ == "__main__":
    
    SAVE_MODEL = True


    n_epochs = 30
    exp_name = init_experiment(f"NDP-batched-sn-None-123S")

    # set seed
    random.seed(2)
    torch.manual_seed(2)
    np.random.seed(2)

    dataset = DWDataset([
        find_file_with_substring("raw_data_2_flyabove_200Hz_80_005_len7636ts_12_iterations.npz"),
        find_file_with_substring("raw_data_2_flyabove_200Hz_80_005_len65911ts_91_iterations.npz"),
        find_file_with_substring("raw_data_3_swapping_200Hz_80_005_len60694ts_100_iterations.npz"),
        find_file_with_substring("raw_data_3_swapping_fast_200Hz_80_005_len56629ts_173_iterations.npz"),
        find_file_with_substring("raw_data_1_flybelow_200Hz_80_005_len68899ts_103_iterations.npz"),
        ])
    
    dataset_eval = DWDataset([find_file_with_substring("raw_data_1_flybelow_200Hz_80_005_len34951ts_51_iterations_testset.npz"),
        find_file_with_substring("raw_data_2_flyabove_200Hz_80_005_len32956ts_46_iterations_testset.npz"),
        find_file_with_substring("raw_data_3_swapping_200Hz_80_005_len29736ts_52_iterations_testset.npz"),
        find_file_with_substring("raw_data_3_swapping_fast_200Hz_80_005_len20802ts_67_iterations_testset.npz"),
        ])

    # Data preparation

   

    train_data, val_data = train_test_split(dataset, test_size=0.25, shuffle=True)

    batch_size = 64


    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    eval_loader = DataLoader(dataset_eval, batch_size=batch_size, shuffle=False)
    
    


    # Model initialization

    model = DWPredictor()
    model = model.to(device)

    # Optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = NormalizedMSELoss()

    # Train the model
    print("Training model...")
    train_errors, val_errors = train_nn(model, train_loader, val_loader, optimizer, criterion, epochs=n_epochs)

    # Evaluate the model
    print("Evaluating model...")
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

    plot_NN_training(train_errors, val_errors)
    model.eval()
    #print(model(st))
    #plot_xy_slices(model)
    #plot_zy_yx_slices(model)
    #plot_z_slices(model)
    #plot_3D_forces(model)

    final_model_name = exp_name + str(n_epochs) + "_eps"  + ".pth"
    if SAVE_MODEL:
        print("Training finished, saving...", final_model_name)
        torch.save(model.state_dict(), final_model_name)