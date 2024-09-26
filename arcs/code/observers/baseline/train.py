import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import DWPredictor
from dataset import DWDataset
import numpy as np

import matplotlib.pyplot as plt

from utils import *


#seed = 0
#torch.manual_seed(seed)
device = "cuda" if torch.cuda.is_available() else "cpu"
n_epochs = 100
lr = 1e-4

    
def train_one_epoch(train_loader, model, optimizer, loss_fn):

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
    


def test(val_loader, model, loss_fn):
    model.eval()
    eval_losses = []

    test_loss, correct = 0,0

    with torch.no_grad():
        # loop over batches of val dataset
        for X,Y in val_loader:

            X, Y = X.to(device), Y.to(device)

            y_pred = model(X)

            eval_losses.append(loss_fn(y_pred, Y).item())
            
    mean_val_error = sum(eval_losses)/len(eval_losses)
    print(f"Mean validation error: {mean_val_error}")
    return mean_val_error


def train():

    # test data:
    # list of tuples: ([px,py,pz, vx,vy,vz], [fx,fy,fz]) 
    # other vehicle - ego vehicle: [0,0,1]->[0,0,-6.5]

    dataset = DWDataset(200)
    train_set, val_set = torch.utils.data.random_split(dataset, [180, 20])

    # init or load model, optimizer and loss 
    model = DWPredictor().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # loss = 
    loss_fn = torch.nn.MSELoss()

    # see initial evaluation before training
    #test_sample = torch.tensor([-1.,1.,0, 0.,0.,0.])
    #st = torch.tensor(test_sample).to(torch.float32).to(device)
    #print("### value at ", test_sample, "is",model(st))

    #plot_xy_slices(model)
    plot_3D_forces(model)

    # setup dataloader
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=20, 
        drop_last=True
    )

    val_loader = DataLoader(
        dataset=val_set,
        batch_size=20, 
        drop_last=True
    )

    # begin training loop

    train_errors, val_errors = [],[]
    for epoch in range(n_epochs):
        print(f"Epoch {epoch+1}\n-------------------------------")
        #if epoch % 100 == 0:
        #    print(f"Epoch {epoch+1} of {n_epochs}")
        
        # train one epoch
        tr_err = train_one_epoch(train_loader, model, optimizer, loss_fn)
        val_err = test(val_loader, model, loss_fn)

        train_errors.append(tr_err)
        val_errors.append(val_err)


        # print model statistics and intermediately save if necessary

    print("Training finished")

    # save or evaluate model if necessary
    plot_NN_training(train_errors, val_errors)
    model.eval()
    #print(model(st))
    plot_xy_slices(model)
    plot_3D_forces(model)





train()