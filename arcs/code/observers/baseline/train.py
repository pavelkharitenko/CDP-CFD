import torch, sys
import torch.nn as nn
from torch.utils.data import DataLoader
from model import DWPredictor
from dataset import DWDataset
import numpy as np
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from arcs.code.utils.utils import *

sys.path.append('../../../../../notify/')
from notify_script_end import notify_ending

SAVE_MODEL = True

load_model = r"C:\Users\admin\Desktop\IDP\CDP-CFD\arcs\code\data_collection\ndp-data-collection\2024-10-05-22-59-25-ndp-model-brilliant-vendor20000_eps.pth"


seed = 0
torch.manual_seed(seed)
device = "cuda" if torch.cuda.is_available() else "cpu"
n_epochs = 20000
lr = 1e-5

    
def train_one_epoch_batch(train_loader, model, optimizer, loss_fn):

    model.train()
    mean_loss = []

    # get batch (x,y)
    for batch, (X,Y) in enumerate(train_loader):
        #print("batch nr:", batch)
        X, Y = X.to(device), Y.to(device)

        # compute forward pass
        y_pred = model(X)

        # compute loss
        loss = loss_fn(y_pred, Y)
        
        # add to statistics: losses.append(loss)
        mean_loss.append(loss.item())

        # update network
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    mean_train_error = sum(mean_loss)/len(mean_loss)
    print(f"Mean train error: {mean_train_error}")
    return mean_train_error
    # print epoch statistics:
    
def train_one_epoch(X_train,Y_train, model, optimizer, loss_fn):

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
    
    
    #print(f"Mean train error: {mean_train_error}")
    return ep_loss
    # print epoch statistics:
    

def test_batch(val_loader, model, loss_fn):
    model.eval()
    eval_losses = []

    

    with torch.no_grad():
        # loop over batches of val dataset
        for X,Y in val_loader:

            X, Y = X.to(device), Y.to(device)

            y_pred = model(X)

            eval_losses.append(loss_fn(y_pred, Y).item())
            
    mean_val_error = sum(eval_losses)/len(eval_losses)
    print(f"Mean validation error: {mean_val_error}")
    return mean_val_error

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
    exp_name = init_experiment("ndp-model")

    dataset = DWDataset(
        r"C:\Users\admin\Desktop\IDP\CDP-CFD\arcs\code\data_collection\ndp-data-collection\2024-10-05-20-26-07-NDP-2-P600-coffee-stream-intermediate-savingsec-15000-ts.p"
    )
    split_ratio = [0.75, 0.25]
    lengths = [int(ratio * len(dataset)) for ratio in split_ratio]
    lengths[-1] = len(dataset) - sum(lengths[:-1])

    x_train, x_val, y_train, y_val = train_test_split(dataset.x, dataset.y, train_size=0.75, test_size=0.25,
                                                      shuffle=True)

    train_set, val_set = torch.utils.data.random_split(dataset, lengths)

    # overfit on small subset
    #x_train = x_train[:20]
    #x_val = x_val[:20]
    #y_train = y_train[:20]
    #y_val = y_val[:20]


    # init or load model, optimizer and loss 
    if load_model:
        model = DWPredictor().to(device)
        model.load_state_dict(torch.load(load_model, weights_only=True))
    else:
        model = DWPredictor().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    loss_fn = torch.nn.MSELoss()

    
    

    #plot_xy_slices(model)
    #plot_3D_forces(model)

    # setup dataloader for minibatch training
    #train_loader = DataLoader(dataset=train_set, batch_size=256, drop_last=True)
    #val_loader = DataLoader(dataset=val_set,batch_size=256, drop_last=True)

    # begin training loop

    train_errors, val_errors = [], []

    for epoch in range(n_epochs):
        #print(f"Epoch {epoch+1}\n-------------------------------")
        #if epoch % 100 == 0:
        #    print(f"Epoch {epoch+1} of {n_epochs}")
        
        # train one epoch with batches
        #tr_err = train_one_epoch_batch(train_loader, model, optimizer, loss_fn)
        epoch += 20000
        tr_err = train_one_epoch(x_train,y_train, model, optimizer, loss_fn)
        

        if epoch % 4000 == 0 and epoch > 0:
            print(f"\nEpoch {epoch+1}\n-------------------------------")
            #val_err = test_batch(val_loader, model, loss_fn)

            val_err = test(x_val, y_val, model, loss_fn)
            train_errors.append(tr_err)
            val_errors.append(val_err)
            print(f'training loss: {train_errors[-1]} \n validation loss: {val_errors[-1]}')
            
            #notify_ending(f'{exp_name} at {epoch}/{n_epochs} ,\n training loss: {train_errors[-1]} \n validation loss: {val_errors[-1]}')

            if SAVE_MODEL:
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
    plot_3D_forces(model)





train()