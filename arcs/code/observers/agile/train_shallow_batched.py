
import torch, sys
from model import AgileShallowPredictor
from dataset import AgileContinousDataset
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
import numpy as np
sys.path.append('../../utils/')

from utils import *

#sys.path.append('../../../../../notify/')
#from notify_script_end import notify_ending

SAVE_MODEL = True
load_model = None

#seed = 123
#torch.manual_seed(seed)

device = "cuda" if torch.cuda.is_available() else "cpu"
n_epochs = 50

lr = 3e-4
#sn_gamma = 4 # scale factor for spectral normalization
sn_gamma = None


model_name = f"Agile-Shallow-batched-sn-{str(sn_gamma)}-123S"

 

def train_nn_with_validation(model, train_loader, val_loader, optimizer, criterion, epochs=20):
    model.train()
    tr_losses = []
    val_losses = []
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
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.6f}")
        
        tr_losses.append(total_loss/len(train_loader))
        val_losses.append(test_nn(model, val_loader, criterion))
    return tr_losses, val_losses



# --- Step 5: Evaluation Function ---
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

    predictions = np.vstack(predictions)
    actuals = np.vstack(actuals)

    mse = mean_squared_error(actuals, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(actuals, predictions)

    print(f"Evaluation Results: RMSE: {rmse:.4f}, R²: {r2:.4f}")
    return actuals, predictions



def train():

    # test data:

    exp_name = init_experiment(model_name)

    dataset = AgileContinousDataset([
        find_file_with_substring("raw_data_2_flyabove_200Hz_80_005_len7636ts_12_iterations.npz"),
        find_file_with_substring("raw_data_2_flyabove_200Hz_80_005_len65911ts_91_iterations.npz"),
        find_file_with_substring("raw_data_3_swapping_200Hz_80_005_len60694ts_100_iterations.npz"),
        find_file_with_substring("raw_data_3_swapping_fast_200Hz_80_005_len56629ts_173_iterations.npz"),
        find_file_with_substring("raw_data_1_flybelow_200Hz_80_005_len68899ts_103_iterations.npz"),
    ])
    
    dataset_eval = AgileContinousDataset([
        find_file_with_substring("raw_data_1_flybelow_200Hz_80_005_len34951ts_51_iterations_testset.npz"),
        find_file_with_substring("raw_data_2_flyabove_200Hz_80_005_len32956ts_46_iterations_testset.npz"),
        find_file_with_substring("raw_data_3_swapping_200Hz_80_005_len29736ts_52_iterations_testset.npz"),
        find_file_with_substring("raw_data_3_swapping_fast_200Hz_80_005_len20802ts_67_iterations_testset.npz"),
    ])

    

    # Data preparation
    # Number of timesteps to look back
    
    


    train_data, val_data = train_test_split(dataset,test_size=0.25, shuffle=True)

    batch_size = 32



    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    eval_loader = DataLoader(dataset_eval, batch_size=batch_size, shuffle=False)

    
    
    

    # overfit on small subset first
    #x_train = x_train[0:-1:1]
    #x_val = x_val[0:-1:1]
    #y_train = y_train[0:-1:1]
    #y_val = y_val[0:-1:1]

    #print("Length of dataset", len(x_train))


    # init or load model, optimizer and loss 
    if load_model:
        model = AgileShallowPredictor().to(device)
        model.load_state_dict(torch.load(load_model, weights_only=True))
    else:
        model = AgileShallowPredictor(output_dim=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    


    train_errors, val_errors = [], []

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    #criterion = torch.nn.MSELoss()
    criterion = WeightedMSELoss()

    # Train the model
    train_errors, val_errors = train_nn_with_validation(model, train_loader, val_loader, optimizer, criterion, epochs=n_epochs)


    # print model statistics and intermediately save if necessary
    final_model_name = exp_name + str(n_epochs) + "_eps"  + ".pth"
    if SAVE_MODEL:
        print("Training finished, saving...", final_model_name)
        torch.save(model.state_dict(), final_model_name)

    #notify_ending(f'FINISHED TRAINING {final_model_name}, \n training loss: {train_errors[-1]} \n validation loss: {val_errors[-1]}')

    
    # save or evaluate model if necessary
    #print(f'\n Final training loss: {train_errors[-1]} \n Final validation loss: {val_errors[-1]}')

    plot_NN_training(train_errors, val_errors)
    #model.eval()
    #print(model(st))
    #(model)
    #plot_zy_yx_slices(model)
    #plot_z_slices(model)
    #plot_3D_forces(model)

    actuals, predictions = evaluate_nn(model, eval_loader)

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(actuals, label="True Forces", linewidth=2)
    plt.plot(predictions, label="Predicted Forces", linewidth=2, linestyle="--")
    plt.xlabel("Time Step")
    plt.ylabel("Force")
    plt.title("True vs Predicted Disturbance Forces")
    plt.legend()
    plt.show()






train()