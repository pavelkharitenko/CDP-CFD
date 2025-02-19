import torch, sys, argparse
from model import AgileShallowPredictor
from dataset import AgileContinousDataset
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
import numpy as np
import os

sys.path.append('../../utils/')
from utils import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--sn_gamma", type=float, default=None, help="SN constant")
    parser.add_argument("--save_model", type=bool, default=False, help="Save trained model")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()

args = parse_args()

# Set seeds for reproducibility
seed = args.seed
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# Use parsed arguments in training script
lr = args.lr
n_epochs = args.epochs
batch_size = args.batch_size
sn_gamma = args.sn_gamma
SAVE_MODEL = args.save_model

device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = f"agl_bs_{batch_size}_sn_{str(sn_gamma)}_lr_{lr}_seed_{seed}_ep{str(n_epochs)}"

def train_nn_with_validation(model, train_loader, val_loader, optimizer, criterion, epochs=20):
    model.train()
    tr_losses = []
    val_losses = []
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

            if sn_gamma:
                apply_spectral_norm(model, sn_gamma)



            total_loss += loss.item()
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.6f}")
        
        tr_losses.append(total_loss/len(train_loader))
        val_losses.append(test_nn(model, val_loader, criterion))
    return tr_losses, val_losses

def test_nn(model, eval_loader, criterion):
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

def evaluate_nn(model, eval_loader):
    model.eval()
    model.to("cpu")
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

def train():
    exp_name = init_experiment(model_name)
    dataset = AgileContinousDataset([
        find_file_with_substring("raw_data_2_flyabove_200Hz_80_005_len7636ts_12_iterations.npz"),
        find_file_with_substring("raw_data_2_flyabove_200Hz_80_005_len65911ts_91_iterations.npz"),
        find_file_with_substring("raw_data_3_swapping_200Hz_80_005_len60694ts_100_iterations.npz"),
        find_file_with_substring("raw_data_3_swapping_fast_200Hz_80_005_len56629ts_173_iterations.npz"),
        find_file_with_substring("raw_data_1_flybelow_200Hz_80_005_len68899ts_103_iterations.npz"),
    ])
    
    # Split dataset with a fixed seed
    train_data, val_data = train_test_split(dataset, test_size=0.25, shuffle=True, random_state=seed)
    
    # seed for the DataLoader
    g = torch.Generator()
    g.manual_seed(seed)

    # Create DataLoader with the generator
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, generator=g)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    if load_model:
        model = AgileShallowPredictor().to(device)
        model.load_state_dict(torch.load(load_model, weights_only=True))
    else:
        model = AgileShallowPredictor(output_dim=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    criterion = WeightedMSELoss()
    train_errors, val_errors = train_nn_with_validation(model, train_loader, val_loader, optimizer, criterion, epochs=n_epochs)

    final_model_name = exp_name + str(n_epochs) + "_eps"  + ".pth"
    if SAVE_MODEL:
        print("Training finished, saving...", final_model_name)
        torch.save(model.state_dict(), final_model_name)

    plot_NN_training(train_errors, val_errors)

train()