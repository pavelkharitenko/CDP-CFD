import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import DWPredictor
from dataset import DWDataset
import numpy as np

import matplotlib.pyplot as plt



device = "cuda" if torch.cuda.is_available() else "cpu"
n_epochs = 1000
lr = 1e-4

def plot_xy_slices(model):

    xy_plane_z_samples = [ -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5]
    xy_range = 1.0
    point_size = 20
    color="autumn_r"
    test_f = []

    fig, ax = plt.subplots(2, len(xy_plane_z_samples), sharex=True, sharey=True, figsize=(6, 6))

    model.eval()
    model.to(device)

    # plot xy-slices
    all_zs = []
    all_imgs = []
    for k, test_z in enumerate(xy_plane_z_samples):
        
        test_xy = np.linspace(start=-xy_range, stop=xy_range, num=point_size)
        test_matrix = np.zeros([point_size**2, 6])
        for i in range(point_size):
            for j in range(point_size):
                test_matrix[i * point_size + j, 0] = test_xy[i]
                test_matrix[i * point_size + j, 1] = test_xy[j]
                test_matrix[i * point_size + j, 2] = test_z
        
        # from Li et al. "Nonlin. MPC for..."
        torch_matrix = torch.from_numpy(test_matrix).to(torch.float32)
        input = torch.autograd.Variable(torch_matrix).cuda()
        output = model(input)
        test_f.append(output)
        zs = output[:, 2]
        
        all_zs.extend(zs.detach().cpu().numpy())

        plot_f = np.zeros((point_size, point_size))
        for i in range(point_size):
            for j in range(point_size):
                plot_f[i, j] = zs[i * point_size + j]

        im = ax[0][k].imshow(plot_f, extent=[-xy_range, xy_range, xy_range, -xy_range], cmap=color)
        all_imgs.append(im)
        ax[0][k].set_title(f"DW at z={test_z}m")
        
    
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)


    # plot xz-slice
    all_ys = []
    for k, test_y in enumerate(xy_plane_z_samples):
        
        test_xz = np.linspace(start=-xy_range, stop=xy_range, num=point_size)
        test_matrix = np.zeros([point_size**2, 6])
        # matrix actually (y,x,z) of state vector:
        for i in range(point_size):
            for j in range(point_size):
                test_matrix[i * point_size + j, 0] = test_y
                test_matrix[i * point_size + j, 1] = test_xz[i]
                test_matrix[i * point_size + j, 2] = test_xz[j]
        
        # from Li et al. "Nonlin. MPC for..."
        torch_matrix = torch.from_numpy(test_matrix).to(torch.float32)
        input = torch.autograd.Variable(torch_matrix).cuda()
        output = model(input)
        test_f.append(output)
        ys = output[:, 1]
        
        all_zs.extend(ys.detach().cpu().numpy())

        plot_f = np.zeros((point_size, point_size))
        for i in range(point_size):
            for j in range(point_size):
                plot_f[i, j] = ys[i * point_size + j]

        im = ax[1][k].imshow(plot_f, extent=[-xy_range, xy_range, xy_range, -xy_range], cmap="autumn_r")
        all_imgs.append(im)
        ax[1][k].set_title(f"DW at y = {test_y}m")

    # add the bar and fix min-max colormap
    for im in all_imgs:
        im.set_clim(vmin=np.min(all_zs), vmax=np.max(all_zs))


    plt.show()


def plot_xy_slice(model):

    xy_plane_z_sample = -1.0
    xy_range = 1.0
    point_size = 200
    color="viridis"
    test_f = []

    fig = plt.figure()
    ax = plt.axes()

    model.eval()
    model.to(device)

    
        
    test_xy = np.linspace(start=-xy_range, stop=xy_range, num=point_size)
    test_matrix = np.zeros([point_size**2, 6])
    for i in range(point_size):
        for j in range(point_size):
            test_matrix[i * point_size + j, 0] = test_xy[i]
            test_matrix[i * point_size + j, 1] = test_xy[j]
            test_matrix[i * point_size + j, 2] = xy_plane_z_sample
        
    torch_matrix = torch.from_numpy(test_matrix).to(torch.float32)
    input = torch.autograd.Variable(torch_matrix).cuda()
    output = model(input)
    test_f.append(output)
    zs = output[:, 2]
    plot_f = np.zeros((point_size, point_size))
    for i in range(point_size):
        for j in range(point_size):
            plot_f[i, j] = zs[i * point_size + j]

    ax.imshow(plot_f,extent=[-xy_range, xy_range, xy_range, -xy_range], cmap=color)
    ax.set_title(f"z={xy_plane_z_sample}m")

    plt.show()


    


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
    # list = [
    # ([px,py,pz, vx,vy,vz], [fx,fy,fz])
    # ]
    # other vehicle - ego vehicle
    # [0,0,1]->[0,0,-6.5]
    dataset = DWDataset(100)
    train_set, val_set = torch.utils.data.random_split(dataset, [90, 10])
    # init or load model, optimizer and loss 
    model = DWPredictor().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # loss = 
    loss_fn = torch.nn.MSELoss()

    # test one sample
    st = torch.tensor(val_set[0][0]).to(torch.float32).to(device)
    print(model(st))

    # setup dataloader
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=12, 
        drop_last=True
    )

    plot_xy_slices(model)


    # begin training
    for epoch in range(n_epochs):
        if epoch % 100 == 0:
            print(f"Epoch {epoch+1} of {n_epochs}")
        
        # train one epoch
        train_one_epoch(train_loader, model, optimizer, loss_fn)

        # print model statistics and intermediately save if necessary

    print("Training finished")
    # save or evaluate model if necessary
    
    model.eval()
    print(model(st))
    plot_xy_slices(model)

   
    



train()