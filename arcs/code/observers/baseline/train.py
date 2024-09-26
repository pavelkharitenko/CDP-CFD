import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import DWPredictor
from dataset import DWDataset
import numpy as np

import matplotlib.pyplot as plt

from utils import plot_xy_slices, plot_3D_forces


#seed = 0
#torch.manual_seed(seed)
device = "cuda" if torch.cuda.is_available() else "cpu"
n_epochs = 1000
lr = 1e-4

    
def train_one_epoch(train_loader, model, optimizer, loss_fn):

    model.train()
    mean_loss = []

    # get batch (x,y)
    for batch, (X,Y) in enumerate(train_loader):
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
    
    #print(f"Mean loss {sum(mean_loss)/len(mean_loss)}")
    

    # optimizer.zero_grad()
    # loss.backward()
    # optimizer.step()

    # print epoch statistics:
    pass


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

    plot_xy_slices(model)
    plot_3D_forces(model)

    # setup dataloader
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=10, 
        drop_last=True
    )

    # begin training loop
    for epoch in range(n_epochs):
        if epoch % 100 == 0:
            print(f"Epoch {epoch+1} of {n_epochs}")
        
        # train one epoch
        train_one_epoch(train_loader, model, optimizer, loss_fn)

        # print model statistics and intermediately save if necessary

    print("Training finished")

    # save or evaluate model if necessary
    model.eval()
    #print(model(st))
    plot_xy_slices(model)
    plot_3D_forces(model)





train()