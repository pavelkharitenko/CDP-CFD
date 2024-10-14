#----------------------------------
# NDP model implemented from Li et al. Nonlinear MPC for Quadrotors in Close-Proximity 
# Flight with Neural Network Downwash Prediction, arXiv:2304.07794v2
# Their implementation found at https://github.com/Li-Jinjie/ndp_nmpc_qd
#----------------------------------
import torch, sys
import torch.nn as nn
from torch.utils.data import DataLoader
from model import NeuralSwarmPredictor
from dataset import DWDataset
import numpy as np
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from utils import *

sys.path.append('../../../../../notify/')
from notify_script_end import notify_ending

SAVE_MODEL = False

#load_model = r"C:\Users\admin\Desktop\IDP\CDP-CFD\arcs\code\observers\ndp\2024-10-10-19-44-57-NDP-Li-Model-sn_scale-4-194k-datapoints-corrected-bias-salty-instructor40000_eps.pth"
load_model = None

seed = 0
torch.manual_seed(seed)
device = "cuda" if torch.cuda.is_available() else "cpu"
n_epochs = 20000
evaluate_at_l_epochs = 100
save_at_m_epochs = 80000

lr = 1e-4
#sn_gamma = 4 # scale factor for spectral normalization
sn_gamma = None
 
def train_one_epoch_with_spectral_normalization(X_train,Y_train, model, optimizer, loss_fn):

    model.train()
    
    

    # get batch (x,y)
    
    #print("batch nr:", batch)
    X, Y = X_train.to(device), Y_train.to(device)

    # compute forward pass
    y_pred = model(X)

    # compute loss
    loss = loss_fn(y_pred, Y)
    
    # add to statistics: losses.append(loss)
    ep_loss = loss.item()

    # update network
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # spectral normalization if provided:
    if sn_gamma:
        for param in model.parameters():
            weights = param.detach()
            if weights.ndim > 1:
                spec_norm = torch.linalg.norm(weights, 2)
                if  spec_norm > sn_gamma:
                    param.data = (param / spec_norm) * sn_gamma
    
    
    #print(f"Mean train error: {mean_train_error}")
    return ep_loss
    # print epoch statistics:
    

def test(X_val,Y_val, model, loss_fn):
    model.eval()
    
    with torch.no_grad():
        X, Y = X_val.to(device), Y_val.to(device)
        y_pred = model(X)
        val_error = loss_fn(y_pred, Y).item()
    return val_error



def train():

    # test data:
    # list of tuples: ([px,py,pz, vx,vy,vz], [fx,fy,fz]) 
    # other vehicle - ego vehicle: [0,0,1]->[0,0,-6.5]

    exp_name = init_experiment(f"NDP-Li-Model-sn_scale-{str(sn_gamma)}-194k-datapoints-corrected-bias")

    dataset = DWDataset([
    #r"datasets\2024-10-06-15-44-22-ndp-2-P600-forgiving-parameter-intermediate-savingsec-35000-ts.p",
    #r"datasets\2024-10-06-17-41-31-ndp-2-P600-wan-osprey-intermediate-savingsec-30000-ts.p",
    #r"datasets\2024-10-07-10-23-45-ndp-2-P600-low-dry-data-intermediate-savingsec-30000-ts.p",
    #r"datasets\2024-10-07-14-10-34-ndp-2-P600-mid-atomic-brass-intermediate-savingsec-35000-ts.p",
    #r"C:\Users\admin\Desktop\IDP\CDP-CFD\arcs\code\data_collection\ndp-data-collection\2024-10-08-15-38-29-Dataset-NDP-2-P600-flush-frank-intermediate-savingsec-30000-ts.p",
    r"C:\Users\admin\Desktop\IDP\CDP-CFD\arcs\code\data_collection\ndp-data-collection\2024-10-08-14-27-23-Dataset-NDP-2-P600-optimal-normal-intermediate-savingsec-25000-ts.p",
    r"C:\Users\admin\Desktop\IDP\CDP-CFD\arcs\code\data_collection\ndp-data-collection\data\two-P600-both-moving-100Hz\2024-10-08-15-38-29-Dataset-NDP-2-P600-flush-frank-720.0sec-72001-ts.p",

    ], from_both_uavs=True)


    x_train, x_val, y_train, y_val = train_test_split(dataset.x, dataset.y, train_size=0.75, test_size=0.25,
                                                      shuffle=True)


    # overfit on small subset first
    #x_train = x_train[:20]
    #x_val = x_val[:20]
    #y_train = y_train[:20]
    #y_val = y_val[:20]


    # init or load model, optimizer and loss 
    if load_model:
        model = NeuralSwarmPredictor().to(device)
        model.load_state_dict(torch.load(load_model, weights_only=True))
    else:
        model = NeuralSwarmPredictor().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    loss_fn = model.loss_fn()


    #plot_xy_slices(model)
    #plot_3D_forces(model)

    train_errors, val_errors = [], []

    # begin training loop
    for epoch in range(n_epochs):
        
        
        tr_err = train_one_epoch_with_spectral_normalization(x_train,y_train, model, optimizer, loss_fn)
        
        # process every n epochs
        if epoch % evaluate_at_l_epochs == 0 and epoch > 0:
            print(f"\nEpoch {epoch+1}\n-------------------------------")
            #val_err = test_batch(val_loader, model, loss_fn)

            val_err = test(x_val, y_val, model, loss_fn)
            train_errors.append(tr_err)
            val_errors.append(val_err)
            print(f'training loss: {train_errors[-1]} \n validation loss: {val_errors[-1]}')
            

        if epoch % save_at_m_epochs == 0:
            if SAVE_MODEL:
                #notify_ending(f'{exp_name} at {epoch}/{n_epochs} ,\n training loss: {train_errors[-1]} \n validation loss: {val_errors[-1]}')
                torch.save(model.state_dict(), exp_name + str(epoch) +"_eps"  + ".pth")
            



        # print model statistics and intermediately save if necessary
    final_model_name = exp_name + str(n_epochs) + "_eps"  + ".pth"
    if SAVE_MODEL:
        print("Training finished, saving...", final_model_name)
        torch.save(model.state_dict(), final_model_name)

    notify_ending(f'FINISHED TRAINING {final_model_name}, \n training loss: {train_errors[-1]} \n validation loss: {val_errors[-1]}')

    
    # save or evaluate model if necessary
    print(f'\n Final training loss: {train_errors[-1]} \n Final validation loss: {val_errors[-1]}')

    plot_NN_training(train_errors, val_errors)
    model.eval()
    #print(model(st))
    #plot_xy_slices(model)
    plot_zy_yx_slices(model)
    plot_z_slices(model)
    plot_3D_forces(model)





train()