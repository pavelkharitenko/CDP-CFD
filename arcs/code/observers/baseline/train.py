import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import DWPredictor
from dataset import DWDataset

device = "cuda" if torch.cuda.is_available() else "cpu"
n_epochs = 1000
lr = 1e-4



# test data:
# list = [
# ([px,py,pz, vx,vy,vz], [fx,fy,fz])
# ]
# ego - other vehicle
# [0,0,-1]->[0,0,-6.5]
dataset = DWDataset(100)



train_set, val_set = torch.utils.data.random_split(dataset, [90, 10])



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
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    print(f"Mean loss {sum(mean_loss)/len(mean_loss)}")
    

    # optimizer.zero_grad()
    # loss.backward()
    # optimizer.step()

    # print epoch statistics:
    pass


def train():
    # init or load model, optimizer and loss 
    model = DWPredictor().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # loss = 
    loss_fn = torch.nn.MSELoss()


    # setup dataloader
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=12, 
        drop_last=True
    )



    for epoch in range(n_epochs):
        print(f"Epoch {epoch+1} of {n_epochs}")
        
        # train one epoch
        train_one_epoch(train_loader, model, optimizer, loss_fn)

        # print model statistics and intermediately save if necessary

    print("Training finished")
    # save or evaluate model if necessary
    
    st = torch.tensor(val_set[0][0]).to(torch.float32).to(device)
    print(model(st))
    #exit(0)

train()