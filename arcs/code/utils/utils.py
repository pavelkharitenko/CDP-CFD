import torch, pickle, randomname, sys, math, os
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from numpy import random as rnd
from shapely.geometry import Polygon, Point
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tabulate import tabulate
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation as R


sys.path.append('../../uav/')
from uav import *

sys.path.append('../../../observers/')


#from ndp.model import DWPredictor
#from SO2.model import ShallowEquivariantPredictor
#from model import DWPredictor


device = "cuda" if torch.cuda.is_available() else "cpu"


# 1 plotting functions

# 1 ndp

def plot_xy_slices_ndp(model):
    """
    Visualize (x,y,z) output of NN as 2D heatmap. Based on "plot_historgrams" at
    https://github.com/Li-Jinjie/ndp_nmpc_qd/blob/master/ndp_nmpc/scripts/dnwash_nn_est/nn_train.py
    """
    
    # Plot xy-slices at different Z-values:
    xy_plane_z_samples = [-0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
    xy_range = 1.0 # size of XY plane
    plane_res = 200 # number of points sampled for plotting in one dimension

    color="autumn_r"
    test_f = []
    fig, ax = plt.subplots(2, len(xy_plane_z_samples), sharex=True, sharey=True, figsize=(6, 6))

    model.eval()
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    
    plotted_forces = [] # save plotted forces for adjusting min and max value of heatmaps

    all_imgs = [] # save all

    # loop over Z-heights and generate plane_res*plane_res sample points
    for idx, z_point in enumerate(xy_plane_z_samples):
        
        xy_samples = np.linspace(start=-xy_range, stop=xy_range, num=plane_res)
        sample_matrix = np.zeros([plane_res**2, 6])

        for i in range(plane_res):
            for j in range(plane_res):
                sample_matrix[i * plane_res + j, 0] = xy_samples[i]
                sample_matrix[i * plane_res + j, 1] = xy_samples[j]
                sample_matrix[i * plane_res + j, 2] = z_point
        
        
        sample_tensor = torch.from_numpy(sample_matrix).to(torch.float32)
        input = torch.autograd.Variable(sample_tensor).cuda()
        output = model(input)
        test_f.append(output)
        zs = output[:, 2]
        
        # add all encountered z-forces and save them for evaluating 
        plotted_forces.extend(zs.detach().cpu().numpy())


        plot_f = np.zeros((plane_res, plane_res))

        for i in range(plane_res):
            for j in range(plane_res):
                plot_f[i, j] = zs[i * plane_res + j]

        im = ax[0][idx].imshow(plot_f, extent=[-xy_range, xy_range, xy_range, -xy_range], cmap=color)
        all_imgs.append(im)
        ax[0][idx].set_title(f"Z = {z_point}m")
        
    
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)


    # plot xz-slice

    for k, test_y in enumerate(xy_plane_z_samples):
        
        test_xz = np.linspace(start=-xy_range, stop=xy_range, num=plane_res)
        test_matrix = np.zeros([plane_res**2, 6])
        # matrix actually (y,x,z) of state vector:
        for i in range(plane_res):
            for j in range(plane_res):
                test_matrix[i * plane_res + j, 0] = test_y
                test_matrix[i * plane_res + j, 1] = test_xz[i]
                test_matrix[i * plane_res + j, 2] = test_xz[j]
        
        
        torch_matrix = torch.from_numpy(test_matrix).to(torch.float32)
        input = torch.autograd.Variable(torch_matrix).cuda()
        output = model(input)
        test_f.append(output)
        ys = output[:, 1]
        
        plotted_forces.extend(ys.detach().cpu().numpy())

        plot_f = np.zeros((plane_res, plane_res))
        for i in range(plane_res):
            for j in range(plane_res):
                plot_f[i, j] = ys[i * plane_res + j]

        im = ax[1][k].imshow(plot_f, extent=[-xy_range, xy_range, xy_range, -xy_range], cmap="autumn_r")
        all_imgs.append(im)
        ax[1][k].set_title(f"Y = {test_y}m")

    # add the bar and fix min-max colormap
    for im in all_imgs:
        im.set_clim(vmin=np.min(plotted_forces), vmax=np.max(plotted_forces))

    fig.suptitle('\n Predicted Downwash Forces acting on Sufferer UAV \n Other drone is Z and Y meters above and right of Sufferer')
    plt.xlabel('Forces in Newton (N)')
    #plt.ylabel('dummy data')
    
    plt.show()

def plot_zy_yx_slices_ndp(model):
    # Plot xy-slices at different Z-values:
    xy_plane_z_samples = [-0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
    xy_range = 1.2 # size of XY plane
    plane_res = 200 # number of points sampled for plotting in one dimension

    color="autumn"
    test_f = []
    fig, ax = plt.subplots(2, len(xy_plane_z_samples), sharex=True, sharey=True, figsize=(6, 6))

    
    model.eval()
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    
    plotted_forces = [] # save plotted forces for adjusting min and max value of heatmaps

    all_imgs = [] # save all

    # loop over Z-heights and generate plane_res*plane_res sample points
    for idx, z_point in enumerate(xy_plane_z_samples):
        
        xy_samples = np.linspace(start=-xy_range, stop=xy_range, num=plane_res)
        sample_matrix = np.zeros([plane_res**2, 6])
        sample_matrix[:,5] = 1.0

        for i in range(plane_res):
            for j in range(plane_res):
                sample_matrix[i * plane_res + j, 0] = xy_samples[i]
                sample_matrix[i * plane_res + j, 1] = xy_samples[j]
                sample_matrix[i * plane_res + j, 2] = z_point
        
        
        sample_tensor = torch.from_numpy(sample_matrix).to(torch.float32)
        input = torch.autograd.Variable(sample_tensor).cuda()
        output = model(input)
        test_f.append(output)
        zs = output[:, 2]
        
        # add all encountered z-forces and save them for evaluating 
        plotted_forces.extend(zs.detach().cpu().numpy())


        plot_f = np.zeros((plane_res, plane_res))

        for i in range(plane_res):
            for j in range(plane_res):
                plot_f[i, j] = zs[i * plane_res + j]

        im = ax[0][idx].imshow(plot_f, extent=[-xy_range, xy_range, xy_range, -xy_range], 
                               cmap=color, origin='lower', interpolation='none')
        all_imgs.append(im)
        ax[0][idx].set_title(f"Z = {z_point}m")
        
    
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    # plot xz-slice

    for k, test_y in enumerate(xy_plane_z_samples):
        
        test_xz = np.linspace(start=-xy_range, stop=xy_range, num=plane_res)
        test_matrix = np.zeros([plane_res**2, 6])
        # matrix actually (y,x,z) of state vector:
        for i in range(plane_res):
            for j in range(plane_res):
                test_matrix[i * plane_res + j, 0] = test_y
                test_matrix[i * plane_res + j, 1] = test_xz[i]
                test_matrix[i * plane_res + j, 2] = test_xz[j]
        
        
        torch_matrix = torch.from_numpy(test_matrix).to(torch.float32)
        input = torch.autograd.Variable(torch_matrix).cuda()
        output = model(input)
        test_f.append(output)
        ys = output[:, 2]
        
        plotted_forces.extend(ys.detach().cpu().numpy())

        plot_f = np.zeros((plane_res, plane_res))
        for i in range(plane_res):
            for j in range(plane_res):
                plot_f[i, j] = ys[i * plane_res + j]

        im = ax[1][k].imshow(plot_f, extent=[-xy_range, xy_range, xy_range, -xy_range], cmap=color,
                             origin='lower', interpolation='none')
        all_imgs.append(im)
        ax[1][k].set_title(f"Y = {test_y}m")

    # add the bar and fix min-max colormap
    for im in all_imgs:
        im.set_clim(vmin=np.min(plotted_forces), vmax=np.max(plotted_forces))

    fig.suptitle('\n Predicted Downwash Forces acting on Sufferer UAV \n Other drone is Z and Y meters above and right of Sufferer')
    plt.xlabel('Forces in Newton (N)')
    #plt.ylabel('dummy data')
    
    plt.show()

def plot_z_slices_ndp(model):
    # Plot xy-slices at different Z-values:
    xy_plane_z_samples = [-0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
    xy_range = 1.0 # size of XY plane
    plane_res = 200 # number of points sampled for plotting in one dimension

    color="autumn"
    test_f = []
    fig, ax = plt.subplots(1, len(xy_plane_z_samples), sharex=True, sharey=True)

    
    model.eval()
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    
    plotted_forces = [] # save plotted forces for adjusting min and max value of heatmaps

    all_imgs = [] # save all

    # loop over Z-heights and generate plane_res*plane_res sample points
    for idx, z_point in enumerate(xy_plane_z_samples):
        
        xy_samples = np.linspace(start=-xy_range, stop=xy_range, num=plane_res)
        sample_matrix = np.zeros([plane_res**2, 6])

        for i in range(plane_res):
            for j in range(plane_res):
                sample_matrix[i * plane_res + j, 0] = xy_samples[i]
                sample_matrix[i * plane_res + j, 1] = xy_samples[j]
                sample_matrix[i * plane_res + j, 2] = z_point
        
        
        sample_tensor = torch.from_numpy(sample_matrix).to(torch.float32)
        input = torch.autograd.Variable(sample_tensor).cuda()
        output = model(input)
        test_f.append(output)
        zs = output[:, 2]
        
        # add all encountered z-forces and save them for evaluating 
        plotted_forces.extend(zs.detach().cpu().numpy())


        plot_f = np.zeros((plane_res, plane_res))

        for i in range(plane_res):
            for j in range(plane_res):
                plot_f[i, j] = zs[i * plane_res + j]

        im = ax[idx].imshow(plot_f, extent=[-xy_range, xy_range, xy_range, -xy_range], 
                               cmap=color, origin='lower', interpolation='none')
        all_imgs.append(im)
        ax[idx].set_title(f"Z = {z_point}m")
        
    
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    # plot xz-slice


    # add the bar and fix min-max colormap
    for im in all_imgs:
        #im.set_clim(vmin=np.min(plotted_forces), vmax=np.max(plotted_forces))
        im.set_clim(vmin=np.min(plotted_forces), vmax=np.max(plotted_forces))


    fig.suptitle('\n Predicted Downwash Forces acting on Sufferer UAV \n Other drone is Z and Y meters above and right of Sufferer')
    plt.xlabel('Forces in Newton (N)')
    #plt.ylabel('dummy data')
    
    plt.show()


def plot_model_compare(model_list):


    # Plot xy-slices at different Z-values:
    xy_plane_z_samples = [-0.5, 0.5, 0.7, 1.0, 1.1, 1.3, 1.5]
    xy_range = 1.0 # size of XY plane
    plane_res = 400 # number of points sampled for plotting in one dimension

    color="autumn"
    test_f = []
    fig, ax = plt.subplots(3, len(xy_plane_z_samples), sharex=True, sharey=True, figsize=(6, 6))
    #ax.text(0, i, harvest[i, j], ha="center", va="center", color="w")
    all_imgs = [] # save all
    plotted_forces = [] # save plotted forces for adjusting min and max value of heatmaps
    xy_samples = np.linspace(start=-xy_range, stop=xy_range, num=plane_res)

    for index, model in enumerate(model_list):
        
        model.eval()
        model.to("cuda" if torch.cuda.is_available() else "cpu")
        


        # loop over Z-heights and generate plane_res*plane_res sample points
        for idx, z_point in enumerate(xy_plane_z_samples):
            
            sample_matrix = np.zeros([plane_res**2, 6])

            for i in range(plane_res):
                for j in range(plane_res):
                    sample_matrix[i * plane_res + j, 0] = xy_samples[i]
                    sample_matrix[i * plane_res + j, 1] = xy_samples[j]
                    sample_matrix[i * plane_res + j, 2] = z_point
            
            
            sample_tensor = torch.from_numpy(sample_matrix).to(torch.float32)
            input = torch.autograd.Variable(sample_tensor).cuda()
            output = model(input)
            test_f.append(output)
            zs = output[:, 2]
            
            # add all encountered z-forces and save them for evaluating 
            plotted_forces.extend(zs.detach().cpu().numpy())


            plot_f = np.zeros((plane_res, plane_res))

            for i in range(plane_res):
                for j in range(plane_res):
                    plot_f[i, j] = zs[i * plane_res + j]

            im = ax[index][idx].imshow(plot_f, extent=[-xy_range, xy_range, xy_range, -xy_range], 
                                cmap=color, origin='lower', interpolation='none')
            all_imgs.append(im)
            ax[index][idx].set_title(f"Z = {z_point}m")
        
        ax[0][0].set_ylabel('S.N.=2')
        ax[1][0].set_ylabel('S.N.=4')
        #ax[2][0].set_ylabel('S.N.=6')
        ax[2][0].set_ylabel('No SN')


    
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    # add the bar and fix min-max colormap
    for im in all_imgs:
        im.set_clim(vmin=np.min(plotted_forces), vmax=np.max(plotted_forces))

    fig.suptitle('\n Predicted Downwash Forces acting on Sufferer UAV \n Other drone is Z and Y meters above and right of Sufferer')
    plt.xlabel('Forces in Newton (N)')
    #plt.ylabel('dummy data')
    
    plt.show()

# 1 so2

def plot_so2_line(model):
    # generate custom trajectory of bravo uav
    model.eval()
    y_range = 1.0
    velocity_y = 0.1
    num_points = 200

    # Generate y positions
    y_positions = np.linspace(-y_range, y_range, num_points)
    uav_2_rel_vectors = []
    dw_predictions = []
    y_positions_list = []

    for i, y in enumerate(y_positions):
        position = [0.0, y, 1.0]
        vB = [0.0, velocity_y, 0.0]
        vA = [0.0, 0.0, 0.0]
        
        state = position + vB + vA
        hx = np.array(h(position, vB, vA))
        with torch.no_grad():
            hx = torch.from_numpy(hx).to(torch.float32).cuda()

            model_pred = model(hx)
            pred_dw = F(state, model_pred)
            dw_predictions.append(pred_dw[2].cpu())

            uav_2_rel_vectors.append(state)
            y_positions_list.append(y)

    plt.plot(y_positions_list, dw_predictions, label="Equivariant prediction")
    plt.legend()
    plt.show()

def plot_xy_slices_so2(model):
    # Plot xy-slices at different Z-values:
    xy_plane_z_samples = [-1.0, -0.5, 0.0, 0.5, 1.0 ]
    xy_range = 1.0 # size of XY plane
    plane_res = 200 # number of points sampled for plotting in one dimension

    color="autumn"
    test_f = []
    fig, ax = plt.subplots(1, len(xy_plane_z_samples), sharex=True, sharey=True)

    
    model.eval()
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    
    plotted_forces = [] # save plotted forces for adjusting min and max value of heatmaps

    all_imgs = [] # save all

    # loop over Z-heights and generate plane_res*plane_res sample points
    for idx, z_point in enumerate(xy_plane_z_samples):
        
        xy_samples = np.linspace(start=-xy_range, stop=xy_range, num=plane_res)
        sample_matrix = np.zeros([plane_res**2, 9])
        sample_matrix[:, 1] = 0.1 # rel velocity 0.1
        plot_f = np.zeros((plane_res, plane_res))

        for i in range(plane_res):
            for j in range(plane_res):
                dx = [xy_samples[i], xy_samples[j], z_point]
                vB = [0.0, 0.1, 0.0]
                vA = [0.0, 0.0, 0.0]

                hx = np.array(h(dx, vB, vA))

                with torch.no_grad():
                    hx = torch.from_numpy(hx).to(torch.float32).cuda()

                    model_pred = model(hx)
                    pred_dw = F(dx, model_pred)
                    
                    plot_f[i, j] = pred_dw[2].cpu()
                    plotted_forces.append(pred_dw[2].cpu().numpy())
               

        im = ax[idx].imshow(plot_f, extent=[-xy_range, xy_range, xy_range, -xy_range], 
                               cmap=color, origin='lower', interpolation='none')
        all_imgs.append(im)
        ax[idx].set_title(f"Z = {z_point}m")
        
    
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    # add the bar and fix min-max colormap
    for im in all_imgs:
        im.set_clim(vmin=np.min(plotted_forces), vmax=np.max(plotted_forces))

    fig.suptitle('\n Predicted Downwash Forces acting on Sufferer UAV \n Other drone is Z and Y meters above and right of Sufferer')
    plt.xlabel('Forces in Newton (N)')
    
    plt.show()

def plot_zy_xy_slices_so2(model):
    # Plot xy-slices at different Z-values:
    xy_plane_z_samples = [-1.0, -0.5, 0.0, 0.5, 1.0 ]
    xy_range = 1.0 # size of XY plane
    plane_res = 200 # number of points sampled for plotting in one dimension

    color="autumn"
    test_f = []
    fig, ax = plt.subplots(2, len(xy_plane_z_samples), sharex=True, sharey=True)

    
    model.eval()
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    
    plotted_forces = [] # save plotted forces for adjusting min and max value of heatmaps

    all_imgs = [] # save all

    # loop over Z-heights and generate plane_res*plane_res sample points
    for idx, z_point in enumerate(xy_plane_z_samples):
        
        xy_samples = np.linspace(start=-xy_range, stop=xy_range, num=plane_res)
        sample_matrix = np.zeros([plane_res**2, 9])
        sample_matrix[:, 1] = 0.1 # rel velocity 0.1
        plot_f = np.zeros((plane_res, plane_res))

        for i in range(plane_res):
            for j in range(plane_res):
                dx = [xy_samples[i], xy_samples[j], z_point]
                vB = [0.0, 0.1, 0.0]
                vA = [0.0, 0.0, 0.0]

                hx = np.array(h(dx, vB, vA))

                with torch.no_grad():
                    hx = torch.from_numpy(hx).to(torch.float32).cuda()

                    model_pred = model(hx)
                    pred_dw = F(dx, model_pred)
                    
                    plot_f[i, j] = pred_dw[2].cpu()
                    plotted_forces.append(pred_dw[2].cpu().numpy())
               

        im = ax[0][idx].imshow(plot_f, extent=[-xy_range, xy_range, xy_range, -xy_range], 
                               cmap=color, origin='lower', interpolation='none')
        all_imgs.append(im)
        ax[0][idx].set_title(f"Z = {z_point}m")
        
    
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    # plot xz-slice

    # loop over Y-heights and generate plane_res*plane_res sample points
    for idx, z_point in enumerate(xy_plane_z_samples):
        
        xy_samples = np.linspace(start=-xy_range, stop=xy_range, num=plane_res)
        sample_matrix = np.zeros([plane_res**2, 9])
        sample_matrix[:, 1] = 0.1 # rel velocity 0.1
        plot_f = np.zeros((plane_res, plane_res))

        for i in range(plane_res):
            for j in range(plane_res):
                dx = [xy_samples[i], z_point, xy_samples[j]]
                vB = [0.0, 0.1, 0.0]
                vA = [0.0, 0.0, 0.0]

                hx = np.array(h(dx, vB, vA))

                with torch.no_grad():
                    hx = torch.from_numpy(hx).to(torch.float32).cuda()

                    model_pred = model(hx)
                    pred_dw = F(dx, model_pred)
                    
                    plot_f[i, j] = pred_dw[2].cpu()
                    plotted_forces.append(pred_dw[2].cpu().numpy())
               

        im = ax[1][idx].imshow(plot_f, extent=[-xy_range, xy_range, xy_range, -xy_range], 
                               cmap=color, origin='lower', interpolation='none')
        all_imgs.append(im)
        ax[1][idx].set_title(f"Y = {z_point}m")

    # add the bar and fix min-max colormap
    for im in all_imgs:
        im.set_clim(vmin=np.min(plotted_forces), vmax=np.max(plotted_forces))

    fig.suptitle('\n Predicted Downwash Forces acting on Sufferer UAV \n Other drone is Z and Y meters above and right of Sufferer')
    plt.xlabel('Forces in Newton (N)')
    
    plt.show()

# 1 ns

def plot_zy_yx_slices_ns(model):
    # Plot xy-slices at different Z-values:
    xy_plane_z_samples = [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.1, 1.3, 1.5]
    xy_range = 1.0 # size of XY plane
    plane_res = 400 # number of points sampled for plotting in one dimension

    color="autumn"
    test_f = []
    fig, ax = plt.subplots(2, len(xy_plane_z_samples), sharex=True, sharey=True, figsize=(6, 6))

    
    model.eval()
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    
    plotted_forces = [] # save plotted forces for adjusting min and max value of heatmaps

    all_imgs = [] # save all

    # loop over Z-heights and generate plane_res*plane_res sample points
    for idx, z_point in enumerate(xy_plane_z_samples):
        
        xy_samples = np.linspace(start=-xy_range, stop=xy_range, num=plane_res)
        sample_matrix = np.zeros([plane_res**2, 6])

        for i in range(plane_res):
            for j in range(plane_res):
                sample_matrix[i * plane_res + j, 0] = xy_samples[i]
                sample_matrix[i * plane_res + j, 1] = xy_samples[j]
                sample_matrix[i * plane_res + j, 2] = z_point
        
        
        sample_tensor = torch.from_numpy(sample_matrix).to(torch.float32)
        input = torch.autograd.Variable(sample_tensor).cuda()
        output = model(input)
        test_f.append(output)
        zs = output
        
        # add all encountered z-forces and save them for evaluating 
        plotted_forces.extend(zs.detach().cpu().numpy())


        plot_f = np.zeros((plane_res, plane_res))

        for i in range(plane_res):
            for j in range(plane_res):
                plot_f[i, j] = zs[i * plane_res + j]

        im = ax[0][idx].imshow(plot_f, extent=[-xy_range, xy_range, xy_range, -xy_range], 
                               cmap=color, origin='lower', interpolation='none')
        all_imgs.append(im)
        ax[0][idx].set_title(f"Z = {z_point}m")
        
    
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    # plot xz-slice

    for k, test_y in enumerate(xy_plane_z_samples):
        
        test_xz = np.linspace(start=-xy_range, stop=xy_range, num=plane_res)
        test_matrix = np.zeros([plane_res**2, 6])
        # matrix actually (y,x,z) of state vector:
        for i in range(plane_res):
            for j in range(plane_res):
                test_matrix[i * plane_res + j, 0] = test_y
                test_matrix[i * plane_res + j, 1] = test_xz[i]
                test_matrix[i * plane_res + j, 2] = test_xz[j]
        
        
        torch_matrix = torch.from_numpy(test_matrix).to(torch.float32)
        input = torch.autograd.Variable(torch_matrix).cuda()
        output = model(input)
        test_f.append(output)
        ys = output
        
        plotted_forces.extend(ys.detach().cpu().numpy())

        plot_f = np.zeros((plane_res, plane_res))
        for i in range(plane_res):
            for j in range(plane_res):
                plot_f[i, j] = ys[i * plane_res + j]

        im = ax[1][k].imshow(plot_f, extent=[-xy_range, xy_range, xy_range, -xy_range], cmap=color,
                             origin='lower', interpolation='none')
        all_imgs.append(im)
        ax[1][k].set_title(f"Y = {test_y}m")

    # add the bar and fix min-max colormap
    for im in all_imgs:
        im.set_clim(vmin=np.min(plotted_forces), vmax=np.max(plotted_forces))

    fig.suptitle('\n Predicted Downwash Forces acting on Sufferer UAV \n Other drone is Z and Y meters above and right of Sufferer')
    plt.xlabel('Forces in Newton (N)')
    #plt.ylabel('dummy data')
    
    plt.show()


def plot_z_slices_ns(model):
    # Plot xy-slices at different Z-values:
    xy_plane_z_samples = [-1.5, -1.0, -0.75, -0.5, 0.0, 0.5, 1.0, 1.25, 1.5]
    xy_range = 1.0 # size of XY plane
    plane_res = 400 # number of points sampled for plotting in one dimension

    color="autumn"
    test_f = []
    fig, ax = plt.subplots(1, len(xy_plane_z_samples), sharex=True, sharey=True)

    
    model.eval()
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    
    plotted_forces = [] # save plotted forces for adjusting min and max value of heatmaps

    all_imgs = [] # save all

    # loop over Z-heights and generate plane_res*plane_res sample points
    for idx, z_point in enumerate(xy_plane_z_samples):
        
        xy_samples = np.linspace(start=-xy_range, stop=xy_range, num=plane_res)
        sample_matrix = np.zeros([plane_res**2, 6])

        for i in range(plane_res):
            for j in range(plane_res):
                sample_matrix[i * plane_res + j, 0] = xy_samples[i]
                sample_matrix[i * plane_res + j, 1] = xy_samples[j]
                sample_matrix[i * plane_res + j, 2] = z_point
        
        
        sample_tensor = torch.from_numpy(sample_matrix).to(torch.float32)
        input = torch.autograd.Variable(sample_tensor).cuda()
        output = model(input)
        test_f.append(output)
        zs = output
        
        # add all encountered z-forces and save them for evaluating 
        plotted_forces.extend(zs.detach().cpu().numpy())


        plot_f = np.zeros((plane_res, plane_res))

        for i in range(plane_res):
            for j in range(plane_res):
                plot_f[i, j] = zs[i * plane_res + j]

        im = ax[idx].imshow(plot_f, extent=[-xy_range, xy_range, xy_range, -xy_range], 
                               cmap=color, origin='lower', interpolation='none')
        all_imgs.append(im)
        ax[idx].set_title(f"Z = {z_point}m")
        
    
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    # add the bar and fix min-max colormap
    for im in all_imgs:
        #im.set_clim(vmin=np.min(plotted_forces), vmax=np.max(plotted_forces))
        im.set_clim(vmin=np.min(plotted_forces), vmax=np.max(plotted_forces))


    fig.suptitle('\n Predicted Downwash Forces acting on Sufferer UAV \n Other drone is Z and Y meters above and right of Sufferer')
    plt.xlabel('Forces in Newton (N)')
    
    
    plt.show()

# other

def plot_zy_xy_slices_empirical(model):
    # Plot xy-slices at different Z-values:
    xy_plane_z_samples = [-1.0, -0.5, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25 ]
    xy_range = 1.0 # size of XY plane
    plane_res = 40 # number of points sampled for plotting in one dimension

    color="autumn"
    test_f = []
    fig, ax = plt.subplots(2, len(xy_plane_z_samples), sharex=True, sharey=True)

    
    
    
    plotted_forces = [] # save plotted forces for adjusting min and max value of heatmaps

    all_imgs = [] # save all

    # loop over Z-heights and generate plane_res*plane_res sample points
    for idx, z_point in enumerate(xy_plane_z_samples):
        
        xy_samples = np.linspace(start=-xy_range, stop=xy_range, num=plane_res)
        sample_matrix = np.zeros([plane_res**2, 9])
        sample_matrix[:, 1] = 0.1 # rel velocity 0.1
        plot_f = np.zeros((plane_res, plane_res))

        for i in range(plane_res):
            for j in range(plane_res):
                dx = [xy_samples[i], xy_samples[j], z_point]

                u2_state = np.zeros(12)
                u1_state = np.pad(dx, (0, 12 - len(dx)), mode='constant', constant_values=0)
                pred_dw = model.F_drag(u1_state, u2_state)[0]
                plot_f[i, j] = pred_dw[2]
                plotted_forces.append(pred_dw[2])
               

        im = ax[0][idx].imshow(plot_f, extent=[-xy_range, xy_range, xy_range, -xy_range], 
                               cmap=color, origin='lower', interpolation='none')
        all_imgs.append(im)
        ax[0][idx].set_title(f"Z = {z_point}m")
        
    
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    # plot xz-slice

    # loop over Y-heights and generate plane_res*plane_res sample points
    for idx, z_point in enumerate(xy_plane_z_samples):
        
        xy_samples = np.linspace(start=-xy_range, stop=xy_range, num=plane_res)
        sample_matrix = np.zeros([plane_res**2, 9])
        sample_matrix[:, 1] = 0.1 # rel velocity 0.1
        plot_f = np.zeros((plane_res, plane_res))

        for i in range(plane_res):
            for j in range(plane_res):
                dx = [xy_samples[i], z_point, xy_samples[j]]
                
                u2_state = np.zeros(12)
                u1_state = np.pad(dx, (0, 12 - len(dx)), mode='constant', constant_values=0)
                pred_dw = model.F_drag(u1_state, u2_state)[0]
                plot_f[i, j] = pred_dw[2]
                plotted_forces.append(pred_dw[2])
               

        im = ax[1][idx].imshow(plot_f, extent=[-xy_range, xy_range, xy_range, -xy_range], 
                               cmap=color, origin='lower', interpolation='none')
        all_imgs.append(im)
        ax[1][idx].set_title(f"Y = {z_point}m")

    # add the bar and fix min-max colormap
    for im in all_imgs:
        im.set_clim(vmin=np.min(plotted_forces), vmax=np.max(plotted_forces))

    fig.suptitle('\n Predicted Downwash Forces acting on Sufferer UAV \n Other drone is Z and Y meters above and right of Sufferer')
    plt.xlabel('Forces in Newton (N)')
    
    plt.show()


def plot_3D_forces(model):

    """
    Visualize (x,y,z) output of NN as 2D heatmap. Based on "plot_historgrams" at
    https://github.com/Li-Jinjie/ndp_nmpc_qd/blob/master/ndp_nmpc/scripts/dnwash_nn_est/nn_train.py
    """
    
    # Plot xy-slices at different Z-values:
    xy_range = 2.0 # size of XY plane
    plane_res = 9 # number of points sampled for plotting in one dimension

    color="autumn_r"
    test_f = []
    #fig, ax = plt.subplots(2, len(xy_plane_z_samples), sharex=True, sharey=True, figsize=(6, 6))

    model.eval()
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    
    plotted_forces = [] 
    all_imgs = []

    # create i x j x k matrix to sample from network forces
    xy_samples = np.linspace(start=-xy_range, stop=xy_range, num=plane_res)
    sample_matrix = np.zeros([plane_res**3, 6])

    for i in range(plane_res):
        for j in range(plane_res):
            for k in range(plane_res):
                sample_matrix[i * plane_res + j + k* plane_res**2, 0] = xy_samples[i]
                sample_matrix[i * plane_res + j + k* plane_res**2, 1] = xy_samples[j]
                sample_matrix[i * plane_res + j + k* plane_res**2, 2] = xy_samples[k]
    
    
    sample_tensor = torch.from_numpy(sample_matrix).to(torch.float32)
    input = torch.autograd.Variable(sample_tensor).cuda()
    xyz_forces = model(input)
    
    # plot result
    input = input.to("cpu").detach().numpy()
    xyz_forces = xyz_forces.to("cpu").detach().numpy()

    plot_3d_vectorfield(input, xyz_forces, 1.0/np.max(np.abs(xyz_forces)), "\n Predicted Downwash Force Vector Field \n (Force length adjusted)")
    

def plot_3d_vectorfield(startpositions, vectors, scale, title):
    # normalize for plotting
    #vectors = vectors / np.linalg.norm(vectors)
    # iterate over input and output and create 3D plot

    
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlabel('X') #, linespacing=4)
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    xs = [xyz[0] for xyz in startpositions]
    ys = [xyz[1] for xyz in startpositions]
    zs = [xyz[2] for xyz in startpositions]
    xmin, xmax = np.min(xs), np.max(xs)
    ymin, ymax = np.min(ys), np.max(ys)
    zmin, zmax = np.min(zs), np.max(zs)

    ax.set_xlim([zmin*1.2,zmax*1.2])
    ax.set_ylim([zmin*1.2,zmax*1.2])
    ax.set_zlim([zmin*1.2,zmax*1.2])

    for i in range(len(startpositions)):
        x,y,z = startpositions[i][:3]
        fx,fy,fz =  vectors[i] * scale # vector length factor for visualization
        ax.quiver(x, y, z, fx, fy, fz, color='steelblue')

    fig.suptitle(title)
    plt.show()
    #plt.savefig("3D_DW_dummydata.png")


def plot_NN_training(train_errors, val_errors, eval_L_epochs=None):
    t = range(len(train_errors))
    plt.plot(t,train_errors, label='training error')
    plt.plot(t,val_errors, label='validation error')
    if eval_L_epochs:
        plt.xlabel("Every " + str(eval_L_epochs) + " epochs")
    else:
        plt.xlabel("epochs")
    plt.ylabel("Errors")
    plt.legend()
    plt.show()


def plot_static_dw_collection(state_sufferer, state_producer, ef_sufferer, jft_sufferer, timestamp_list):
    fig, axes = plt.subplots(2, 2)
    color1 = 'tab:red'
    color2 = 'tab:brown'
    color3 = 'tab:cyan'
    color4 = 'tab:pink'
    color5 = 'tab:olive'
    color6 = 'tab:purple'

    # Plot Both drones position
    ax1 = axes[0][0]
    ax1.set_title('DW Producer & Sufferer Positions in the YZ-Plane')
    ax1.scatter([xyz[1] for xyz in state_producer], [xyz[2] for xyz in state_producer], label='Producer UAV')
    ax1.scatter([xyz[1] for xyz in state_sufferer], [xyz[2] for xyz in state_sufferer], label='Sufferer UAV')
    ax1.set_xlabel('Y-Position (m)')
    ax1.set_ylabel('Z-Position (m)')
    ax1.legend()
    
    # Plot recorded forces over time
    ax2 = axes[1][0]
    ax2.set_title('External Forces on Sufferer UAV, EF and JFT sensors')
    ax2.set_xlabel('Y-Position (m) of Producer')
    ax2.set_ylabel('Force (N)')
    ax2.plot([xyz[1] for xyz in state_producer], [xyz[0] for xyz in jft_sufferer], label='x force (jft)')
    ax2.plot([xyz[1] for xyz in state_producer], [xyz[1] for xyz in jft_sufferer], label='y force (jft)')
    ax2.plot([xyz[1] for xyz in state_producer], [xyz[2] for xyz in jft_sufferer], label='z force (jft)')
    ax2.plot([xyz[1] for xyz in state_sufferer], ef_sufferer, label='z force (ef)')
    ax2.legend()

    ax3 = axes[0][1]
    ax3.set_title('DW Producer & Sufferer Positions in the XY-Plane')
    ax3.scatter([xyz[1] for xyz in state_producer], [xyz[0] for xyz in state_producer], label='Producer UAV')
    ax3.scatter([xyz[1] for xyz in state_sufferer], [xyz[0] for xyz in state_sufferer], label='Sufferer UAV')
    ax3.set_xlabel('Y-Position (m)')
    ax3.set_ylabel('X-Position (m)')
    ax3.legend()

    
    ax4 = axes[1][1]
    ax4.set_title('ExtForce vs JointForTor sensor Z force comparison')
    ax4.set_xlabel('time (s)')
    ax4.set_ylabel('Force (N)')
    ax4.plot(timestamp_list, ef_sufferer, label='ExtForSen', color=color4)
    ax4.plot(timestamp_list, [xyz[2] for xyz in jft_sufferer], label='JointForTor', color=color5)
    ax4.legend()




# 2 experiment 

def init_experiment(env_name):  
    experiment_name = datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "-" + env_name + "-"+ randomname.get_name()
    print("###############################################")
    print("#### -- Beginning Experiment '", experiment_name, "' -- ####")
    return experiment_name


def save_experiment(exp_name, uav_list, success, sim_duration, bias=None):
    exp_name += "-" + str(sim_duration) + "sec-" + str(len(uav_list[0].timestamp_list)) + "-ts"
    
    if not success:
        #print("#### --- during simulation, an error occured, not saving results of experiment", exp_name)
        exit(1)
    else:
        #print("#### Saving experiment", exp_name, "...")
       
        exp_obj = {'exp_name': exp_name, 'uav_list': uav_list, 'bias': bias}
        
            

        save_name = exp_name + '.p'
        with open(save_name, 'wb') as handle:
            pickle.dump(exp_obj, handle)
        

        #print("#### Saved.")
        #print("################################")
        return save_name


def load_forces_from_dataset(exp_pkl_path):
    with open(exp_pkl_path, 'rb') as handle:
            experiment = pickle.load(handle)
    print("Loaded " + experiment["exp_name"])
    return experiment

def extract_labeled_dataset_ndp(uav_list, bias=None):
    uav_1, uav_2 = uav_list

    title = f"\n (relative from {uav_2.name}'s view: rel_state = {uav_1.name} - {uav_2.name})"
    rel_states = np.array(uav_1.states) - np.array(uav_2.states)
    rel_state_list = [rel_state[0:6] for rel_state in rel_states]
    dw_force_vectors = []
        
    if not uav_2.mounted_jft_sensor:
        print("Deriving forces from residual formula")
        dw_force_vectors = compute_residual_dw_forces(uav_2)
    else:
        # extract forces from UAV2's mounted jft sensor
        if bias:
            dw_force_vectors = np.array([(0,0,-(dw_force[2]-bias)) for dw_force in uav_2.mounted_jft_sensor_list])

        else:
            dw_force_vectors = np.array([(0,0,-dw_force[2]) for dw_force in uav_2.mounted_jft_sensor_list])

    xy_data = list(zip(rel_state_list, dw_force_vectors))
    print("lenghts inside rel state list:", len(rel_state_list))
    print("lenghts inside dw force vectors:", len(dw_force_vectors))
    
    #print("data extracted:", xy_data[-2200:-2000])
    #print("positive data", [pos_data for pos_data in xy_data if pos_data[0][2]<-1.0])

    
    plot_3d_vectorfield(rel_state_list[0::20], dw_force_vectors[0::20], 1.0/np.max(np.abs(dw_force_vectors)),  "Created z-force dataset from collected data" + title)
        
    return rel_state_list, dw_force_vectors

def extract_labeled_dataset_so2(uav_list):
    """
    Extract relative position (uav1 - uav2), uav2 vel, uav1 vel, and dw forces.
    """
    uav_1, uav_2 = uav_list

    rel_states = np.array(uav_1.states) - np.array(uav_2.states)
    rel_pos = [rel_state[0:3] for rel_state in rel_states]
    v_a_list = [state[3:6] for state in np.array(uav_1.states)]
    v_b_list = [state[3:6] for state in np.array(uav_2.states)]

    dw_force_vectors = compute_residual_dw_forces(uav_2)

    return rel_pos, v_b_list, v_a_list, dw_force_vectors
   

def extract_labeled_dataset_ns(uav_list, bias=None):
    uav_1, uav_2 = uav_list

    title = f"\n (relative from {uav_2.name}'s view: rel_state = {uav_1.name} -  {uav_2.name})"
    rel_states = np.array(uav_1.states) - np.array(uav_2.states)
    rel_state_list = [rel_state[0:6] for rel_state in rel_states]
    dw_force_vectors = []
        
    print("Deriving forces from residual formula")
    dw_force_vectors = compute_residual_dw_forces(uav_2)
    dw_force_vectors = [f_xyz[2] for f_xyz in dw_force_vectors]

    xy_data = list(zip(rel_state_list, dw_force_vectors))
    print("lenghts inside rel state list:", len(rel_state_list))
    print("lenghts inside dw force vectors:", len(dw_force_vectors))
    
    #print("data extracted:", xy_data[-2200:-2000])
    #print("positive data", [pos_data for pos_data in xy_data if pos_data[0][2]<-1.0])

    
    #plot_3d_vectorfield(rel_state_list[0::20], dw_force_vectors[0::20], 1.0/np.max(np.abs(dw_force_vectors)),  "Created z-force dataset from collected data" + title)
        
    return rel_state_list, dw_force_vectors

    
def compute_residual_dw_forces(uav):
    """
    Compute from F_uav = -(mg+bias) + Fu + Fd     => Fd = F_uav + (mg+bias) - Fu 

    Returns list of (0,0,fz) tuples
    """
    Fz_total = [uav.total_mass * state[8] for state in uav.states] # recorded actual uav m*a z-Force
    Fg = [uav.total_mass * 9.81 for state in uav.states] # uav gravitational force m*g
    Fz_u_total = np.array([-np.sum(body_r1_r2_r3_r4, axis=0) for body_r1_r2_r3_r4 in uav.jft_forces_list])
    g_bias_steady_state = 4.5 # on average, Fu compentates always more by 5N, so it is not disturbance force
    
    #print("total thrust forces:", Fz_u_total[-500:-400])
    
    dw_forces = np.array(Fz_total) + np.array(Fg) - np.array(Fz_u_total)
    dw_forces += g_bias_steady_state
    #print("dw forces raw:", dw_forces[-500:-400])
    #print("dw forces after subtracting bias:", dw_forces[-500:-400])

    dw_force_vectors = np.array([(0,0,dw_force) for dw_force in dw_forces])

    return dw_force_vectors


# planning
    
def plan_next_coords(sample_distance, safety_distance, self_pos, other_uav_pos):
    xmax, xmin = other_uav_pos[0] + sample_distance, other_uav_pos[0] - sample_distance
    ymax, ymin = other_uav_pos[1] + sample_distance, other_uav_pos[1] - sample_distance
    zmax, zmin = other_uav_pos[2] + sample_distance, other_uav_pos[2] - sample_distance

    #print("minmax", xmax, xmin)
    #print("minmax", ymax, ymin)
    #print("minmax", zmax, zmin)

    xmax_safety, xmin_safety = other_uav_pos[0] + safety_distance, other_uav_pos[0] - safety_distance
    ymax_safety, ymin_safety = other_uav_pos[1] + safety_distance, other_uav_pos[1] - safety_distance
    zmax_safety, zmin_safety = other_uav_pos[2] + safety_distance, other_uav_pos[2] - safety_distance

    #print("minmax safety", xmax_safety, xmin_safety)
    #print("minmax safety", ymax_safety, ymin_safety)
    #print("minmax safety", zmax_safety, zmin_safety)

    curr_x, curr_y, curr_z = self_pos

    for i in range(200):
        new_x, new_y, new_z = rnd.uniform(xmin, xmax), rnd.uniform(ymin, ymax), rnd.uniform(zmin, zmax)
        next_coords = rnd.choice(['xy','xz','yz'])

        if next_coords == 'xy':
            if zmax_safety >= curr_z and curr_z >= zmin_safety:
                continue 
            else:
                return new_x, new_y, curr_z
        
        if next_coords == 'xz':
            if ymax_safety >= curr_y and curr_y >= ymin_safety:
                continue 
            else:
                return new_x, curr_y, new_z
        
        if next_coords == 'yz':
            if xmax_safety >= curr_x and curr_x >= xmin_safety:
                continue 
            else:
                return curr_x, new_y, new_z
    
    return other_uav_pos[0], other_uav_pos[1], other_uav_pos[2] + 1


def sample_next_coords(sample_distance_z, sample_distance, safety_distance, other_uav_pos):
    xmax, xmin = other_uav_pos[0] + sample_distance, other_uav_pos[0] - sample_distance
    ymax, ymin = other_uav_pos[1] + sample_distance, other_uav_pos[1] - sample_distance
    zmax, zmin = other_uav_pos[2] + sample_distance_z, other_uav_pos[2] + safety_distance
    #print("minmax", xmax, xmin)
    #print("minmax", ymax, ymin)
    #print("minmax", zmax, zmin)

    new_x, new_y, new_z = rnd.uniform(xmin, xmax), rnd.uniform(ymin, ymax), rnd.uniform(zmin, zmax)
        
    return new_x, new_y, new_z


def fly_circle(time, radius, frequency, z_pos):
    
    nan = float('NaN')
    px = radius * np.sin(2.0 * np.pi * frequency * time) 
    py = radius - radius * np.cos(2.0 * np.pi * frequency * time)
    return (0.0, px, py, z_pos, nan, nan, nan, nan, nan, nan, 0.0, nan)
    
def sample_circle_coords(zmax, zmin, radius_max, radius_min, other_uav_pos):
     new_z, new_radius = rnd.uniform(zmin + other_uav_pos[2], zmax + other_uav_pos[2]), rnd.uniform(radius_min, radius_max)

     return new_z, new_radius


def evaluate_zy_force_curvature(models, rel_state_vectors):
    """
    Plots zy-force figure for recorded flyby experiement
    """
    predicted_z_curves = []
    for index, model in enumerate(models):
        dw_forces = model.evaluate(rel_state_vectors)
        predicted_z_curves.append(dw_forces)
  
    return predicted_z_curves



def sample_3d_point(y_pos):
    """
    Samples a 3D point from the specified Gaussian distributions with clamping.
    Returns:
        (x, y, z): A tuple of sampled 3D coordinates.
    """
    y_pos = np.round(y_pos,1)
    # Y-coordinate: Two peaks at -0.7 and 0.7, clamped between [-1.0, -0.4] and [0.4, 1.0]
    if y_pos > 0.0:  # Choose one of the two peaks
        y = np.random.normal(loc=-1.5, scale=0.15)
        y = np.clip(y, -1.7, -0.7)
    else:
        y = np.random.normal(loc=1.5, scale=0.15)
        y = np.clip(y, 0.7, 1.7)

    # Z-coordinate: Gaussian centered at 0.75, clamped between [0.0, 1.5]
    z = np.random.normal(loc=0.75, scale=0.25)
    z = np.clip(z, 0.0, 1.5)

    # X-coordinate: Gaussian centered at 0, clamped between [-0.3, 0.3]
    x = np.random.normal(loc=0, scale=0.15)
    x = np.clip(x, -0.4, 0.4)

    return x, y, z

def plot_distribution(num_samples=100):
    """
    Generates and plots the 3D distribution of sampled points.
    Args:
        num_samples: Number of points to sample for visualization.
    """
    points = np.array([sample_3d_point(1.0) for _ in range(num_samples)])
    x, y, z = points[:, 0], points[:, 1], points[:, 2]

    # Create 3D scatter plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(x, y, z, c=z, cmap='viridis', s=5, alpha=0.7)

    # Labeling and aesthetics
    ax.set_title("3D Gaussian Distribution Sampling", fontsize=14)
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Z-axis")
    fig.colorbar(scatter, label="Z-coordinate", shrink=0.6)

    plt.show()


def smooth_with_savgol(data, window_size=5, poly_order=2):
    """
    Smooths the input data using the Savitzky-Golay filter while keeping the number of output points unchanged.
    
    Args:
        data: List or array of measurements to be smoothed.
        window_size: Size of the sliding window (must be odd and >= poly_order + 2).
        poly_order: Order of the polynomial used for fitting.
        
    Returns:
        Smoothed data with the same length as the input.
    """
    # Ensure window_size is odd and valid
    if window_size % 2 == 0:
        raise ValueError("window_size must be an odd number.")
    if window_size < poly_order + 2:
        raise ValueError("window_size is too small for the given poly_order.")

    # Apply Savitzky-Golay filter
    smoothed_data = savgol_filter(data, window_length=window_size, polyorder=poly_order)
    return smoothed_data

def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

def sample_from_range(lower,upper):
    random_numbers = np.random.uniform(lower, upper, size=1)

    return random_numbers[0]


# UAV state and force computations

def extract_and_plot_data(data_set_paths=None, 
                          predictors=[], 
                          time_seq=None, uav_1_states=None, uav_2_states=None, 
                          plot=False, roll_iterations=True, save=False):
    """
    After collecting agile flight data, plot, evaluate and save uav states as dataset.
    """
    if data_set_paths:
        uav_1_states = []
        uav_2_states = []
        time_seq = []

        for path in data_set_paths:
            exp = load_forces_from_dataset(path)
            uav_1, uav_2 = exp['uav_list']
            uav_1_states.extend(uav_1.states[0:])
            uav_2_states.extend(uav_2.states[0:])
            time_seq.extend(uav_1_states.timestamp_list)
    
    else: 
        uav_1_states = uav_1_states
        uav_2_states = uav_2_states
        time_seq = time_seq

    plot = True
    mass = 3.035
    g = -9.85
    

    # 1 extract necessary parameters from uav states
    rel_state_vector_list = np.array(uav_1_states) - np.array(uav_2_states)
    u2_rotations = [R.from_euler('xyz', [yaw_pitch_roll[2], yaw_pitch_roll[1], yaw_pitch_roll[0]], degrees=False) for yaw_pitch_roll in np.array(uav_2_states)[:,9:12]]
    u2_avg_rps = np.mean(np.abs(np.array(uav_2_states)[:,22:26]), axis=1)
    u2_rps_rot = zip(u2_avg_rps, u2_rotations)

    
    overlap_indices = np.where(np.abs(np.array(uav_1_states)[:,1] - np.array(uav_2_states)[:,1]) < 10.6)[0]
    overlaps = np.array(time_seq)[overlap_indices]

    # 2 compute uav actual forces, smooth them, and compute controller z-axis forces, and their residual disturbance
    u2_accelerations = np.array(uav_2_states)[:,8]
    u2_z_forces = u2_accelerations * mass
    smoothed_u2_z_forces = smooth_with_savgol(u2_z_forces, window_size=21, poly_order=1)
    u2_thrusts = [u2_rotations.apply([0, 0, rps_to_thrust_p005_mrv80(avg_rps)])[2] + (g*mass) for (avg_rps, u2_rotations) in u2_rps_rot]
    u2_z_dw_forces = smoothed_u2_z_forces - u2_thrusts

    
    model_paths = [
        #r"C:\Users\admin\Desktop\IDP\CDP-CFD\arcs\code\observers\ndp\2024-11-20-13-11-05-NDP-predictor-sn_scale-4-300k-ts-flyby-navy-sill20000_eps.pth",
        #r"C:\Users\admin\Desktop\IDP\CDP-CFD\arcs\code\observers\SO2\2024-11-20-00-10-30-SO2-Model-below-sn_scale-None-gray-javelin20000_eps.pth",
        #r"C:\Users\admin\Desktop\IDP\CDP-CFD\arcs\code\observers\SO2\2024-11-20-12-30-17-SO2-Model-below-sn_scale-4-dull-flow20000_eps.pth",
    ]
    models = []

    # colors= ["green", "yellow", "red"]

    # model = DWPredictor()
    # model.load_state_dict(torch.load(model_paths[0], weights_only=True))
    # models.append(model)

    # model = ShallowEquivariantPredictor()
    # model.load_state_dict(torch.load(model_paths[1], weights_only=True))
    # models.append(model)

    # model = ShallowEquivariantPredictor()
    # model.load_state_dict(torch.load(model_paths[2], weights_only=True))
    # models.append(model)


    # predictions = evaluate_zy_force_curvature(models, np.array(rel_state_vector_list)[:,:6])
    # labels = [ 
    #     "NDP new data SN<4 ", 
    #     "SO2-Equiv.", 
    #     "SO2-Equiv. SN<4", 
    # ]



    # # 3 plot if needed
    # if plot:
    #     fig = plt.subplot()
    #     #fig.plot(u2_z_forces, label="UAV's z-axis forces") # unsmoothed
    #     plot_array_with_segments(fig, time_seq, smoothed_u2_z_forces, color="blue", roll=roll_iterations, label="UAV total Z-forces")
    #     plot_array_with_segments(fig, time_seq, u2_thrusts,  color="orange", roll=roll_iterations, label="controller z-forces")
    #     plot_array_with_segments(fig, time_seq, u2_z_dw_forces, color="magenta", roll=roll_iterations, label="downwash disturbance forces", overlaps=overlaps)
        
    #     # evaluate predictors
    #     for idx, prediction in enumerate(predictions):
    #         plot_array_with_segments(fig, time_seq, prediction[:,2], roll=roll_iterations, color=colors[idx], label=labels[idx])
    #         compute_rmse(prediction[:,2][overlap_indices], u2_z_dw_forces[overlap_indices],label=labels[idx])
            
        
        



    #     plt.ylabel("Force [N]")
    #     plt.xlabel("time [s]")
    #     plt.grid()
    #     plt.title("Actual bottom UAV Z-forces, controller's thrust, and residual downwash force")
    #     plt.legend()
    #     plt.show()



    # 4 create label:
    u2_z_dw_forces = np.array([[0,0,z_force] for z_force in u2_z_dw_forces])

    # label format v1: [state_uav1, state_uav2, dw_forces]
    # find indices where the time array decreases (reset points) for number of iterations
    reset_indices = np.where(np.diff(time_seq) < 0)[0] + 1  # Add 1 to shift to the start of the next segment
    reset_indices = np.append([0], reset_indices)  # Include the start of the array as the first segment boundary
    reset_indices = np.append(reset_indices, len(time_seq))  # Include the end of the array as the last boundary
    n_itrs = len(reset_indices)
    if save:
        np.savez(f"raw_data_3_swapping_fast_200Hz_80_005_len{len(uav_1_states)}ts_{n_itrs}_iterations", 
                 uav_1_states=uav_1_states, 
                 uav_2_states=uav_2_states, 
                 dw_forces=u2_z_dw_forces)


def extract_data(uav_1_states, uav_2_states, time_seq):
    mass = 3.035
    g = -9.85

    # 1 extract necessary parameters from uav states
    u2_rotations = [R.from_euler('xyz', [yaw_pitch_roll[2], yaw_pitch_roll[1], yaw_pitch_roll[0]], degrees=False) for yaw_pitch_roll in np.array(uav_2_states)[:,9:12]]
    u2_avg_rps = np.mean(np.abs(np.array(uav_2_states)[:,22:26]), axis=1)
    u2_rps_rot = zip(u2_avg_rps, u2_rotations)


    overlap_indices = np.where(np.abs(np.array(uav_1_states)[:,1] - np.array(uav_2_states)[:,1]) < 3.5)[0]
    overlaps = np.array(time_seq)[overlap_indices]

    # 2 compute uav actual forces, smooth them, and compute controller z-axis forces, and their residual disturbance
    u2_accelerations = np.array(uav_2_states)[:,8]
    u2_z_forces = u2_accelerations * mass
    smoothed_u2_z_forces = smooth_with_savgol(u2_z_forces, window_size=21, poly_order=1)
    u2_thrusts = [u2_rotations.apply([0, 0, rps_to_thrust_p005_mrv80(avg_rps)])[2] + (g*mass) for (avg_rps, u2_rotations) in u2_rps_rot]
    u2_z_dw_forces = smoothed_u2_z_forces - u2_thrusts

    return u2_z_dw_forces, overlap_indices, overlaps

def extract_dw_forces(uav_states):
    mass = 3.035
    g = -9.85

    uav_states = np.array(uav_states)

    # 1 extract necessary parameters from uav states
    u2_rotations = [R.from_euler('xyz', [yaw_pitch_roll[2], yaw_pitch_roll[1], yaw_pitch_roll[0]], degrees=False) for yaw_pitch_roll in uav_states[:,9:12]]
    u2_avg_rps = np.mean(np.abs(uav_states[:,22:26]), axis=1)
    

    # 2 compute uav actual forces, smooth them, and compute controller z-axis forces, and their residual disturbance
    smoothed_u2_x_forces, smoothed_u2_y_forces, smoothed_u2_z_forces = extract_uav_forces(uav_states)

    u2_x_thrusts = [u2_rotations.apply([0, 0, rps_to_thrust_p005_mrv80(avg_rps)])[0] for (avg_rps, u2_rotations) in zip(u2_avg_rps, u2_rotations)]
    u2_y_thrusts = [u2_rotations.apply([0, 0, rps_to_thrust_p005_mrv80(avg_rps)])[1] for (avg_rps, u2_rotations) in zip(u2_avg_rps, u2_rotations)]
    u2_z_thrusts = [u2_rotations.apply([0, 0, rps_to_thrust_p005_mrv80(avg_rps)])[2] + (g*mass) for (avg_rps, u2_rotations) in zip(u2_avg_rps, u2_rotations)]
    #print(u2_x_thrusts[:30])
    u2_x_dw_forces = smoothed_u2_x_forces - np.array(u2_x_thrusts)
    u2_y_dw_forces = smoothed_u2_y_forces - np.array(u2_y_thrusts)
    u2_z_dw_forces = smoothed_u2_z_forces - np.array(u2_z_thrusts)

    return u2_x_dw_forces, u2_y_dw_forces, u2_z_dw_forces


def extract_uav_forces(uav_states):
    mass = 3.035
    # compute uav actual forces, smooth them, and compute controller z-axis forces, and their residual disturbance
    uav_x_forces = uav_states[:,6] * mass
    uav_y_forces = uav_states[:,7] * mass
    uav_z_forces = uav_states[:,8] * mass

    smoothed_uav_x_forces = smooth_with_savgol(uav_x_forces, window_size=21, poly_order=1)
    smoothed_uav_y_forces = smooth_with_savgol(uav_y_forces, window_size=21, poly_order=1)
    smoothed_uav_z_forces = smooth_with_savgol(uav_z_forces, window_size=21, poly_order=1)

    return smoothed_uav_x_forces, smoothed_uav_y_forces, smoothed_uav_z_forces


def rps_to_thrust_p005_mrv80(mean_rps):
    """
    Function to calculate thrust for given avg. rps, accurate for 340-430rps.
    Constants derived from range of measurements
    """
    a = 0.00019339212
    b = 0.01897496901
    c = -4.52623347271

    #Ct =  0.000362
    #rho = 1.225
    #A = 0.11948
    #k = Ct * rho * A
    #d = 0.3
    #F = k * one_rotor_rps**2
    
    return a*mean_rps**2 + b* mean_rps + c


def omega_to_thrust_p005_mrv80(mean_rps):
    """
    Computes force by F = c * omega^2, accurate for any range
    Constants derived experimentally.
    """
    Ct =  0.000362
    rho = 1.225
    A = 0.11948
    k = Ct * rho * A
    d = 0.3

    return 4.0*k * mean_rps**2.0

def discretize_shapes(vertices_list, n_cells=18.0, plot=False):
    """
    Creates 2d frame from vertices, disretizes as grid with cells and plots optionally
    """

    total_grid_points = []

    if plot:
        
        plt.figure(figsize=(8, 8))

    for vertices in vertices_list:
        frame_shape = Polygon(vertices)

        # Set the grid size (5 cm = 0.05 meters)

        # Define the bounding box for the grid based on the diamond shape
        min_x, min_y, max_x, max_y = frame_shape.bounds

        x_cell_size = (max_x - min_x) / n_cells
        y_cell_size = (max_y - min_y) / n_cells


        # Generate the grid points within the bounding box
        x_coords = np.arange(min_x, max_x , x_cell_size)
        y_coords = np.arange(min_y, max_y, y_cell_size)
        
        X, Y = np.meshgrid(x_coords, y_coords)
        grid_points = np.vstack([X.ravel(), Y.ravel()]).T

        # Check which points are inside the diamond shape
        inside_points = [point for point in grid_points if frame_shape.contains(Point(point))]

        # Convert the inside points to an array for plotting
        inside_points = np.array(inside_points)
        
        
        total_grid_points.extend(inside_points)

        # Plot the diamond shape and the discretized points
        if plot:
            #plt.figure(figsize=(8, 8))
            plt.plot(*zip(*vertices, vertices[0]), color='black', linewidth=2)
            plt.scatter(inside_points[:, 0], inside_points[:, 1], color='blue', s=4)
    
    total_grid_points = np.array(total_grid_points)

    if plot:
        plt.xlabel("X (meters)")
        plt.ylabel("Y (meters)")
        plt.legend()
        plt.title(f"Discretized Frame for {x_cell_size}m grid cell length")
        plt.axis("equal")
        plt.grid(True)
        #plt.scatter(total_grid_points[:,0], total_grid_points[:,1], color="red", s=5)
        plt.show()
    
    A_cell = x_cell_size*y_cell_size
    return total_grid_points, x_cell_size, y_cell_size



def compute_thrust_vector(uav_states):
        """
        Returns list of Thrust vectors [x,y,z] rotated accordingly to uav's orientation
        """
        
        thrusts = rps_to_thrust_p005_mrv80(np.mean(np.abs(uav_states[:,22:26]), axis=1, keepdims=True))
        #print(thrusts[:5])
        thrust_vectors = np.column_stack((np.zeros(len(thrusts)), np.zeros(len(thrusts)), thrusts))
        rotations = R.from_euler('zyx', uav_states[:,9:12], degrees=False)
        thrust_list = rotations.apply(thrust_vectors) 

        return thrust_list



def plot_array_with_segments(fig, time, array, roll=True, color=None, label=None, overlaps=[]):
    """
    Plots a single numpy array either fully or segmented where the time array resets to 0.
    
    Args:
        time (np.ndarray): The time array with periodic resets to 0.
        array (np.ndarray): The numpy array to be plotted against the time array.
        plot_fully (bool): If True, plots the entire array; if False, plots segments starting at resets.
    """
    if len(time) != len(array):
        raise ValueError("The time array and data array must have the same length.")

    if not roll:
        
        #fig.plot(time, array, label=label, color=color)
        fig.plot(array, label=label, color=color)

        
            
        #fig.axvline(overlaps, color='grey', linestyle='--', alpha=0.1)


    else:

        # Find the indices where the time array decreases (reset points)
        reset_indices = np.where(np.diff(time) < 0)[0] + 1  # Add 1 to shift to the start of the next segment
        reset_indices = np.append([0], reset_indices)  # Include the start of the array as the first segment boundary
        reset_indices = np.append(reset_indices, len(time))  # Include the end of the array as the last boundary
        
        for i in range(len(reset_indices) - 1):
            start, end = reset_indices[i], reset_indices[i + 1]
            segment_time = time[start:end]
            segment_data = array[start:end]
            fig.plot(segment_time, segment_data, color=color)
    
        for line_time in overlaps:
            if line_time in time:  # Ensure the line time exists in the time array
                fig.axvline(line_time, color='grey', linestyle='--', alpha=0.1)



def compute_rmse(array1, array2, label=None):
    rmse = np.sqrt(np.mean((array1 - array2) ** 2))
    print("length of comparison", len(array1))
    if label:
        print(f"RMSE of {label} is: {rmse}")
    else:
        print(f"RMSE is", str(rmse))
    return rmse

def plot_uav_angles(time, roll, pitch, yaw, title):
    """
    Plots the roll, pitch, and yaw angles of a UAV over time.
    
    Args:
        time (np.ndarray): Time array.
        roll (np.ndarray): Roll angle array in degrees or radians.
        pitch (np.ndarray): Pitch angle array in degrees or radians.
        yaw (np.ndarray): Yaw angle array in degrees or radians.
    """
    if not (len(time) == len(roll) == len(pitch) == len(yaw)):
        raise ValueError("Time, roll, pitch, and yaw arrays must have the same length.")
    
    plt.figure(figsize=(12, 6))
    # Plot roll
    plt.plot(time, roll, label="Roll", color="blue", linewidth=1.5)
    
    # Plot pitch
    plt.plot(time, pitch, label="Pitch", color="green", linewidth=1.5)
    
    # Plot yaw
    plt.plot(time, yaw, label="Yaw", color="red", linewidth=1.5)

    # Labels and grid
    plt.xlabel("Time (s)")
    plt.ylabel("Angle (degrees)")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    #plt.show()


def compute_torques(r1, r2, r3, r4):
    # r1 forward left, r3 forward right, r4 back left, r2 back right
    Ct =  0.000362
    rho = 1.225
    A = 0.11948
    k = Ct * rho * A
    d = 0.3


    # roll torque (τ_x)
    tau_x = k * d * (r2**2 + r3**2 - r1**2 - r4**2)
    # pitch torque (τ_y)
    tau_y = k * d * (-r1**2 - r3**2 + r2**2 + r4**2) # forward tilt is positive
    # yaw torque (τ_z)
    tau_z = -k * d*(r1**2 - r3**2 + r2**2 - r4**2)

    return tau_x, tau_y, tau_z


def find_file_with_substring(substring):
    """
    Search for a file containing the given substring in its name within the current directory and subdirectories.
    
    Args:
        substring (str): The substring to look for in the file name.
    
    Returns:
        str: The full path of the file if found, or None if not found.
    """
    root_dir = "../../../../" # Set the root directory to the current working directory
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if substring in filename:
                return os.path.abspath(os.path.join(dirpath, filename))
    return None


def compute_metrics(y_true, y_pred):
    """
    Compute RMSE, NRMSE, MAE, and R2 score between true and predicted values.
    Args:
        y_true (array): Ground truth values.
        y_pred (array): Predicted values.
    Returns:
        dict: A dictionary with RMSE, NRMSE, MAE, and R2 score.
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    #nrmse_range = rmse / (np.max(y_true) - np.min(y_true))  # Range-normalized RMSE
    nrmse_range = rmse / np.mean(y_true)  # Range-normalized RMSE

    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {
        "RMSE": rmse,
        "NRMSE (Mean)": nrmse_range,
        "MAE": mae,
        "R2 Score": r2
    }



def compute_weighted_metrics(results, sample_sizes):
    """
    Compute weighted RMSE, MAE, and R2 score across datasets.
    Args:
        results (list of dicts): List of metrics per dataset.
        sample_sizes (list): Number of samples for each dataset.
    Returns:
        dict: Weighted RMSE, MAE, and R2 score.
    """
    total_samples = sum(sample_sizes)

    # Weighted RMSE: Combine squared errors
    weighted_rmse = np.sqrt(
        sum((result["RMSE"] ** 2) * n for result, n in zip(results, sample_sizes)) / total_samples
    )

    # Weighted MAE: Combine absolute errors
    weighted_mae = sum(result["MAE"] * n for result, n in zip(results, sample_sizes)) / total_samples

    # Weighted R2: Average the R2 scores, weighted by sample sizes
    weighted_r2 = sum(result["R2 Score"] * n for result, n in zip(results, sample_sizes)) / total_samples

    return {
        "Weighted RMSE": weighted_rmse,
        "Weighted MAE": weighted_mae,
        "Weighted R2": weighted_r2
    }


def analyze_and_plot_forces(avg_rps_list, measured_a_zs, mass, rps_to_thrust_func, ignore_first_k=200, window_size=20):
    """
    Analyze and plot UAV forces and control inputs. TODO: consider correct attitude of controllers thrust vector

    :param avg_rps_list: List of average RPS values.
    :param measured_a_zs: List of measured accelerations along the z-axis.
    :param mass: Mass of the UAV.
    :param rps_to_thrust_func: Function to compute thrust from RPS.
    :param ignore_first_k: Number of initial data points to ignore for analysis.
    :param window_size: Window size for running average and variance calculation.
    """
    # Compute thrusts and UAV forces
    thrusts = [rps_to_thrust_func(avg_rps) - 9.81 * mass for avg_rps in avg_rps_list]
    uav_z_forces = np.array(measured_a_zs) * mass

    # Plot UAV forces and controller forces
    plt.figure(figsize=(10, 5))
    plt.plot(moving_average(uav_z_forces, window_size), label="UAV's z-axis forces")
    plt.plot(thrusts, label="Controller's RPS forces")
    plt.legend()
    plt.show()

    # Analyze data after ignoring initial points
    data = uav_z_forces[ignore_first_k:]
    running_avg = np.convolve(data, np.ones(window_size) / window_size, mode='valid')
    running_var = [np.var(data[i:i + window_size]) for i in range(len(data) - window_size + 1)]
    running_std_dev = np.sqrt(running_var)

    # Compute total variance and standard deviation
    total_var = np.full(len(running_avg), np.var(running_avg))
    total_std = np.sqrt(total_var)

    # Define x-axis for the metrics
    x_vals = np.arange(window_size - 1, len(data))

    # Plot running metrics and variance tube
    plt.figure(figsize=(12, 6))
    plt.plot(data, label="IMU Measurements", linewidth=0.4)
    plt.fill_between(x_vals, running_avg - total_std, running_avg + total_std, 
                     color='orange', alpha=0.4, label="Variance (±1 STD)")
    plt.plot(x_vals, running_avg, color='orange', label="Running Average", linewidth=2)

    # Adjust thrusts length for comparison
    thrusts = thrusts[ignore_first_k:]
    plt.plot(x_vals, thrusts[window_size-1:len(data)], label="Controller's Input Thrust", 
             linewidth=2, color="magenta")

    # Labels, title, and legend
    plt.xlabel("Timesteps")
    plt.ylabel("Value")
    plt.title("Raw IMU Measurement and Running Average for Smoother Measurements")
    plt.legend()
    plt.show()

def plot_trajectory_analysis(actual_positions, planned_positions, actual_velocities):    
    """
    Used to plot z-tracking error and velocities of one uav.
    """
    sns.set(style="whitegrid")

    # Calculate errors and RMSE
    position_errors = planned_positions[:, :3] - actual_positions
    velocity_errors = planned_positions[:, 3:6] - actual_velocities
    
    avg_position_errors = np.mean(np.abs(position_errors), axis=0)
    avg_velocity_errors = np.mean(np.abs(velocity_errors), axis=0)
    
    radial_position_errors = np.sqrt(position_errors[:, 0]**2 + position_errors[:, 1]**2)
    avg_radial_position_error = np.mean(radial_position_errors)

    fig, axes = plt.subplots(3, 2, figsize=(18, 15))
    
    # Z-Position Plot
    sns.lineplot(x=np.arange(len(actual_positions)), y=actual_positions[:, 2], ax=axes[0, 0], label='Actual Z', color='blue')
    sns.lineplot(x=np.arange(len(planned_positions)), y=planned_positions[:, 2], ax=axes[0, 0], label='Target Z', color='green')
    axes[0, 0].set_title('Z-Position')
    axes[0, 0].set_ylabel('Z Position (m)')
    
    # Z-Error Plot
    sns.lineplot(x=np.arange(len(position_errors)), y=position_errors[:, 2], ax=axes[1, 0], label='Z-Error', color='red')
    axes[1, 0].axhline(avg_position_errors[2], linestyle='--', color='orange', label=f'Avg Error ({avg_position_errors[2]:.2f})')
    axes[1, 0].set_title('Z-Position Error')
    axes[1, 0].set_ylabel('Error (m)')
    axes[1, 0].legend()
    
    # XYZ Velocities Plot
    for i, label in enumerate(['X', 'Y', 'Z']):
        sns.lineplot(x=np.arange(len(actual_velocities)), y=actual_velocities[:, i], ax=axes[0, 1], label=f'{label}-Velocity', color=['blue', 'green', 'red'][i])
    axes[0, 1].set_title('XYZ Velocities')
    axes[0, 1].set_ylabel('Velocity (m/s)')
    axes[0, 1].legend()
    
    # XYZ Velocity Error Plot
    for i, label in enumerate(['X', 'Y', 'Z']):
        sns.lineplot(x=np.arange(len(velocity_errors)), y=velocity_errors[:, i], ax=axes[1, 1], label=f'{label}-Velocity Error', color=['blue', 'green', 'red'][i])
    axes[1, 1].axhline(avg_velocity_errors[i], linestyle='--', color='orange', label=f'Avg Error ({avg_velocity_errors[i]:.2f})')
    axes[1, 1].set_title('XYZ Velocity Errors')
    axes[1, 1].set_ylabel('Error (m/s)')
    axes[1, 1].legend()
    
    # Statistics Bar Plot for Position Errors
    axes[2, 0].bar(['X Error', 'Y Error', 'Z Error', 'Radial Error'], 
                    [avg_position_errors[0], avg_position_errors[1], avg_position_errors[2], avg_radial_position_error], 
                    color=['blue', 'green', 'red', 'orange'])
    axes[2, 0].set_title('Average Position Errors')
    axes[2, 0].set_ylabel('Error (m)')
    
    # Statistics Bar Plot for Velocity Errors
    axes[2, 1].bar(['X Velocity Error', 'Y Velocity Error', 'Z Velocity Error'], 
                    avg_velocity_errors, color=['blue', 'green', 'red'])
    axes[2, 1].set_title('Average Velocity Errors')
    axes[2, 1].set_ylabel('Error (m/s)')

    plt.tight_layout()
    plt.show()




def compute_trajectory_errors(actual_positions, actual_velocities, planned_positions, ignore_start=0, ignore_end=0):
    """
    Compute errors between the actual trajectory and interpolated planned trajectory.
    """
    if ignore_end > 0:
        actual_positions = actual_positions[ignore_start:-ignore_end]
        actual_velocities = actual_velocities[ignore_start:-ignore_end]
        planned_positions = planned_positions[ignore_start:-ignore_end]
    else:
        actual_positions = actual_positions[ignore_start:]
        actual_velocities = actual_velocities[ignore_start:]
        planned_positions = planned_positions[ignore_start:]
    
    time_actual = np.arange(len(actual_positions))
    time_planned = np.linspace(0, len(actual_positions) - 1, len(planned_positions))
    
    interp_func = interp1d(time_planned, planned_positions, axis=0, kind='linear', fill_value='extrapolate')
    interpolated_planned_positions = interp_func(time_actual)
    
    position_errors = actual_positions - interpolated_planned_positions[:, :3]
    velocity_errors = actual_velocities - interpolated_planned_positions[:, 3:6]
    
    return position_errors, velocity_errors, interpolated_planned_positions

def plot_trajectory_analysis_two_uavs(actual_positions_uav1, planned_positions_uav1, actual_velocities_uav1, 
                                      actual_positions_uav2, planned_positions_uav2, actual_velocities_uav2, 
                                      ignore_start=0, ignore_end=0, feedforward=[]):
    """
    Used to plot tracking errors, velocities, and relative distances of two UAVs.
    """
    overlap_threshold = 1.0
    sns.set(style="whitegrid")
    plt.rcParams.update({'font.size': 10})  # Smaller font size

    # Compute errors for UAV1 and UAV2
    pos_err_uav1, vel_err_uav1, interp_planned_uav1 = compute_trajectory_errors(actual_positions_uav1, actual_velocities_uav1, planned_positions_uav1, ignore_start, ignore_end)
    pos_err_uav2, vel_err_uav2, interp_planned_uav2 = compute_trajectory_errors(actual_positions_uav2, actual_velocities_uav2, planned_positions_uav2, ignore_start, ignore_end)
    
    # Calculate relative XY distance between UAV1 and UAV2
    rel_xy_distances = np.sqrt((actual_positions_uav1[:, 0] - actual_positions_uav2[:, 0])**2 + 
                               (actual_positions_uav1[:, 1] - actual_positions_uav2[:, 1])**2)
    rel_xy_distances = rel_xy_distances[ignore_start:len(rel_xy_distances)-ignore_end] if ignore_end > 0 else rel_xy_distances[ignore_start:]
    overlap_indices = np.where(rel_xy_distances < overlap_threshold)[0]

    actual_positions_uav1 = actual_positions_uav1[ignore_start:]
    actual_positions_uav2 = actual_positions_uav2[ignore_start:]
    actual_velocities_uav1 = actual_velocities_uav1[ignore_start:]
    actual_velocities_uav2 = actual_velocities_uav2[ignore_start:]


    if len(overlap_indices) == 0:
        overlap_indices = [len(rel_xy_distances)-1]

    fig, axes = plt.subplots(3, 2, figsize=(18, 25))

    # Z-Position Plot for UAV1 and UAV2
    sns.lineplot(x=np.arange(len(actual_positions_uav1)), y=actual_positions_uav1[:, 2], ax=axes[0, 0], label='UAV1 Actual Z', color='blue')
    sns.lineplot(x=np.arange(len(interp_planned_uav1)), y=interp_planned_uav1[:, 2], ax=axes[0, 0], label='UAV1 Planned Z', color='green')
    sns.lineplot(x=np.arange(len(actual_positions_uav2)), y=actual_positions_uav2[:, 2], ax=axes[0, 0], label='UAV2 Actual Z', color='orange')
    sns.lineplot(x=np.arange(len(interp_planned_uav2)), y=interp_planned_uav2[:, 2], ax=axes[0, 0], label='UAV2 Planned Z', color='purple')
    axes[0, 0].set_title('Z-Position')
    axes[0, 0].set_ylabel('Z Position (m)')
    for index in overlap_indices:
        axes[0, 0].axvline(index, color='red', linestyle='--', alpha=0.3)

    # Z-Error Plot for UAV1 and UAV2
    sns.lineplot(x=np.arange(len(pos_err_uav1)), y=pos_err_uav1[:, 2], ax=axes[1, 0], label='UAV1 Z-Err.', color='blue')
    sns.lineplot(x=np.arange(len(pos_err_uav2)), y=pos_err_uav2[:, 2], ax=axes[1, 0], label='UAV2 Z-Err.', color='orange')
    axes[1, 0].set_title('Z-Position Error')
    axes[1, 0].set_ylabel('Error (m)')
    axes[1, 0].legend()
    
    # Relative XY Distance Plot
    sns.lineplot(x=np.arange(len(rel_xy_distances)), y=rel_xy_distances, ax=axes[2, 0], label='Rel. XY Dist.', color='cyan')
    # plot additional info
    if len(feedforward) > 0:
        feedforward = feedforward[ignore_start:]
        sns.lineplot(x=np.arange(len(feedforward)), y=feedforward*0.1, ax=axes[2, 0], label='predictors Z-force', color='magenta')
    

    axes[2, 0].set_title('Relative XY Distance between UAV1 and UAV2')
    axes[2, 0].set_ylabel('Distance (m) and Feedforward forces from predictor')
    for index in overlap_indices:
        axes[2, 0].axvline(index, color='red', linestyle='--', alpha=0.3)

    # XYZ Velocities Time Series Plot for UAV1 and UAV2
    for i, label in enumerate(['X', 'Y', 'Z']):
        sns.lineplot(x=np.arange(len(actual_velocities_uav1)), y=actual_velocities_uav1[:, i], ax=axes[0, 1], label=f'UAV1 {label}-Velocity', color=['blue', 'green', 'red'][i], linestyle='-')
        sns.lineplot(x=np.arange(len(actual_velocities_uav2)), y=actual_velocities_uav2[:, i], ax=axes[0, 1], label=f'UAV2 {label}-Velocity', color=['cyan', 'magenta', 'yellow'][i], linestyle='--')
    axes[0, 1].set_title('UAV1 and UAV2 XYZ Velocities')
    axes[0, 1].set_ylabel('Velocity (m/s)')
    axes[0, 1].legend()

    # Compute statistics for position and velocity errors
    avg_pos_errors_uav1 = np.mean(np.abs(pos_err_uav1), axis=0)
    avg_pos_errors_uav2 = np.mean(np.abs(pos_err_uav2), axis=0)
    avg_vel_errors_uav1 = np.mean(np.abs(vel_err_uav1), axis=0)
    avg_vel_errors_uav2 = np.mean(np.abs(vel_err_uav2), axis=0)
    
    max_pos_errors_uav1 = np.max(np.abs(pos_err_uav1), axis=0)
    max_pos_errors_uav2 = np.max(np.abs(pos_err_uav2), axis=0)
    max_vel_errors_uav1 = np.max(np.abs(vel_err_uav1), axis=0)
    max_vel_errors_uav2 = np.max(np.abs(vel_err_uav2), axis=0)

    avg_overlap_pos_errors_uav1 = np.mean(np.abs(pos_err_uav1[overlap_indices]), axis=0)
    avg_overlap_pos_errors_uav2 = np.mean(np.abs(pos_err_uav2[overlap_indices]), axis=0)
    avg_overlap_vel_errors_uav1 = np.mean(np.abs(vel_err_uav1[overlap_indices]), axis=0)
    avg_overlap_vel_errors_uav2 = np.mean(np.abs(vel_err_uav2[overlap_indices]), axis=0)

    max_overlap_pos_errors_uav1 = np.max(np.abs(pos_err_uav1[overlap_indices]), axis=0)
    max_overlap_pos_errors_uav2 = np.max(np.abs(pos_err_uav2[overlap_indices]), axis=0)
    max_overlap_vel_errors_uav1 = np.max(np.abs(vel_err_uav1[overlap_indices]), axis=0)
    max_overlap_vel_errors_uav2 = np.max(np.abs(vel_err_uav2[overlap_indices]), axis=0)

    # Colors corresponding to axes
    colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'orange']

    # Bar plots for position errors
    x_labels = ['U1 Px err.', 'U1 Py Err.', 'U1 Pz err.', 'U2 Px err.', 'U2 Py err.', 'U2 Pz err.']
    avg_errors = np.concatenate((avg_pos_errors_uav1, avg_pos_errors_uav2))
    avg_overlap_errors = np.concatenate((avg_overlap_pos_errors_uav1, avg_overlap_pos_errors_uav2))
    max_errors = np.concatenate((max_pos_errors_uav1, max_pos_errors_uav2))
    max_overlap_errors = np.concatenate((max_overlap_pos_errors_uav1, max_overlap_pos_errors_uav2))

    x = np.arange(len(x_labels))
    width = 0.2  # Width of individual bars in triplets

    for i in range(len(x_labels)):
        axes[1, 1].bar(x[i] - width, avg_errors[i], width, label='Avg Error' if i == 0 else "", color=colors[i % 4])
        axes[1, 1].bar(x[i] - 0.5*width, max_errors[i], width, label='Max Error' if i == 0 else "", color=colors[i % 4])
        axes[1, 1].bar(x[i] + 0.5* width, avg_overlap_errors[i], width, label='Avg Overlap Error' if i == 0 else "", color=colors[i % 4])
        axes[1, 1].bar(x[i] + width, max_overlap_errors[i], width, label='Max Overlap' if i == 0 else "", color=colors[i % 4])

        
    axes[1, 1].set_title('Position Errors: Avg., Max., Avg (overlap) Max. (overlap)')
    axes[1, 1].set_ylabel('Error (m)')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(x_labels)
    

    # Bar plots for velocity errors
    x_labels = ['U1 Vx err.', 'U1 Vy err.', 'U1 Vz err.', 'U2 Vx err.', 'U2 Vy err.', 'U2 Vz err.']
    avg_errors = np.concatenate((avg_vel_errors_uav1, avg_vel_errors_uav2))
    max_errors = np.concatenate((max_vel_errors_uav1, max_vel_errors_uav2))
    avg_overlap_errors = np.concatenate((avg_overlap_vel_errors_uav1, avg_overlap_vel_errors_uav2))
    max_overlap_errors = np.concatenate((max_overlap_vel_errors_uav1, max_overlap_vel_errors_uav2))

    x = np.arange(len(x_labels))

    for i in range(len(x_labels)):
        axes[2, 1].bar(x[i] - width, avg_errors[i], width, label='Avg Error' if i == 0 else "", color=colors[i % 4])
        axes[2, 1].bar(x[i] -0.5* width, max_errors[i], width, label='Max Error' if i == 0 else "", color=colors[i % 4])
        axes[2, 1].bar(x[i] + 0.5* width, avg_overlap_errors[i], width, label='Avg Overlap Error' if i == 0 else "", color=colors[i % 4])
        axes[2, 1].bar(x[i] + width, max_overlap_errors[i], width, label='Max Overlap Error' if i == 0 else "", color=colors[i % 4])

        
    axes[2, 1].set_title('Velocity Errors: Avg., Max., Avg (overlap) Max. (overlap)')
    axes[2, 1].set_ylabel('Error (m/s)')
    axes[2, 1].set_xticks(x)
    axes[2, 1].set_xticklabels(x_labels)
    

    plt.tight_layout()
    plt.show()



def plot_trajectory_analysis_rmse(
    actual_positions_uav1, planned_positions_uav1, actual_velocities_uav1,
    actual_positions_uav2, planned_positions_uav2, actual_velocities_uav2,
    ignore_start=0, ignore_end=0
):
    """
    Plot trajectory tracking errors, velocities, and relative distances for two UAVs.
    Also computes RMSE for positions and velocities.
    """
    overlap_threshold = 1.0
    sns.set(style="whitegrid")
    plt.rcParams.update({'font.size': 10})

    # Compute position and velocity errors for both UAVs
    pos_err_uav1 = actual_positions_uav1 - planned_positions_uav1[:,:3]
    vel_err_uav1 = actual_velocities_uav1 - planned_positions_uav1[:,3:6]
    pos_err_uav2 = actual_positions_uav2 - planned_positions_uav2[:,:3]
    vel_err_uav2 = actual_velocities_uav2 - planned_positions_uav2[:,3:6]

    # Ignore start and end sections for analysis
    pos_err_uav1 = pos_err_uav1[ignore_start:len(pos_err_uav1) - ignore_end]
    vel_err_uav1 = vel_err_uav1[ignore_start:len(vel_err_uav1) - ignore_end]
    pos_err_uav2 = pos_err_uav2[ignore_start:len(pos_err_uav2) - ignore_end]
    vel_err_uav2 = vel_err_uav2[ignore_start:len(vel_err_uav2) - ignore_end]

    # Compute RMSE for position and velocity trajectories
    rmse_pos_uav1 = np.sqrt(np.mean(np.square(pos_err_uav1), axis=0))
    rmse_vel_uav1 = np.sqrt(np.mean(np.square(vel_err_uav1), axis=0))
    rmse_pos_uav2 = np.sqrt(np.mean(np.square(pos_err_uav2), axis=0))
    rmse_vel_uav2 = np.sqrt(np.mean(np.square(vel_err_uav2), axis=0))

    # Print RMSE results
    print("UAV 1 - RMSE (Position): X: {:.3f}, Y: {:.3f}, Z: {:.3f}".format(*rmse_pos_uav1))
    print("UAV 1 - RMSE (Velocity): X: {:.3f}, Y: {:.3f}, Z: {:.3f}".format(*rmse_vel_uav1))
    print("UAV 2 - RMSE (Position): X: {:.3f}, Y: {:.3f}, Z: {:.3f}".format(*rmse_pos_uav2))
    print("UAV 2 - RMSE (Velocity): X: {:.3f}, Y: {:.3f}, Z: {:.3f}".format(*rmse_vel_uav2))

    # Relative XY distance between UAVs
    rel_xy_distances = np.sqrt((actual_positions_uav1[:, 0] - actual_positions_uav2[:, 0])**2 +
                               (actual_positions_uav1[:, 1] - actual_positions_uav2[:, 1])**2)
    rel_xy_distances = rel_xy_distances[ignore_start:len(rel_xy_distances) - ignore_end]

    # Plot position, velocity, and relative distance errors
    fig, axes = plt.subplots(3, 2, figsize=(18, 20))

    # UAV1 and UAV2 Z-Position
    sns.lineplot(y=actual_positions_uav1[:, 2], x=np.arange(len(actual_positions_uav1)), ax=axes[0, 0], label='UAV1 Actual Z')
    sns.lineplot(y=planned_positions_uav1[:, 2], x=np.arange(len(planned_positions_uav1)), ax=axes[0, 0], label='UAV1 Planned Z')
    sns.lineplot(y=actual_positions_uav2[:, 2], x=np.arange(len(actual_positions_uav2)), ax=axes[0, 0], label='UAV2 Actual Z', linestyle="--")
    sns.lineplot(y=planned_positions_uav2[:, 2], x=np.arange(len(planned_positions_uav2)), ax=axes[0, 0], label='UAV2 Planned Z', linestyle="--")
    axes[0, 0].set_title("Z-Position Comparison")
    axes[0, 0].set_ylabel("Z-Position (m)")

    # UAV1 and UAV2 Z-Error
    sns.lineplot(y=pos_err_uav1[:, 2], x=np.arange(len(pos_err_uav1)), ax=axes[1, 0], label='UAV1 Z-Error')
    sns.lineplot(y=pos_err_uav2[:, 2], x=np.arange(len(pos_err_uav2)), ax=axes[1, 0], label='UAV2 Z-Error')
    axes[1, 0].set_title("Z-Position Error")
    axes[1, 0].set_ylabel("Error (m)")

    # Relative XY distance
    sns.lineplot(y=rel_xy_distances, x=np.arange(len(rel_xy_distances)), ax=axes[2, 0], label="Relative XY Distance")
    axes[2, 0].set_title("Relative XY Distance Between UAV1 and UAV2")
    axes[2, 0].set_ylabel("Distance (m)")

    # Velocity components for UAV1 and UAV2
    for i, label in enumerate(['X', 'Y', 'Z']):
        sns.lineplot(y=actual_velocities_uav1[:, i], x=np.arange(len(actual_velocities_uav1)), ax=axes[0, 1], label=f'UAV1 {label}-Velocity')
        sns.lineplot(y=actual_velocities_uav2[:, i], x=np.arange(len(actual_velocities_uav2)), ax=axes[0, 1], label=f'UAV2 {label}-Velocity', linestyle="--")
    axes[0, 1].set_title("Velocity Components")
    axes[0, 1].set_ylabel("Velocity (m/s)")

    # Bar plots for RMSE
    x_labels = ['X', 'Y', 'Z']
    rmse_pos = [rmse_pos_uav1, rmse_pos_uav2]
    rmse_vel = [rmse_vel_uav1, rmse_vel_uav2]

    for i, label in enumerate(["UAV1", "UAV2"]):
        axes[1, 1].bar(np.arange(3) + i * 0.3, rmse_pos[i], width=0.3, label=f"{label} Position RMSE")
        axes[2, 1].bar(np.arange(3) + i * 0.3, rmse_vel[i], width=0.3, label=f"{label} Velocity RMSE")

    axes[1, 1].set_title("Position RMSE")
    axes[1, 1].set_ylabel("RMSE (m)")
    axes[1, 1].set_xticks(np.arange(3))
    axes[1, 1].set_xticklabels(x_labels)

    axes[2, 1].set_title("Velocity RMSE")
    axes[2, 1].set_ylabel("RMSE (m/s)")
    axes[2, 1].set_xticks(np.arange(3))
    axes[2, 1].set_xticklabels(x_labels)

    plt.tight_layout()
    plt.show()


def analyze_forces(uav_forces, thrust_forces, predictor_forces_z, nfc_forces, dt):
    """
    Plots two subplots with y-data and calculated x-data based on time step.
    - dt: time step in seconds
    """

    # 1) display uav z-force, controller z-force, residual z-force (downwash), and predictors z-force

    
    #print(thrust_forces)
    uav_x_forces = [force[0] for force in uav_forces]
    uav_y_forces = [force[1] for force in uav_forces]
    uav_z_forces = [force[2] for force in uav_forces]

    uav_x_thrust = np.array([thrust[0] for thrust in thrust_forces])
    uav_y_thrust = np.array([thrust[1] for thrust in thrust_forces])
    uav_z_thrust = np.array([thrust[2] for thrust in thrust_forces])
    uav_z_thrust -= 9.81*3.3035

    smoothed_uav_x_forces = smooth_with_savgol(uav_x_forces, window_size=31, poly_order=1)
    smoothed_uav_y_forces = smooth_with_savgol(uav_y_forces, window_size=31, poly_order=1)
    smoothed_uav_z_forces = smooth_with_savgol(uav_z_forces, window_size=31, poly_order=1)

    uav_x_dw_forces = smoothed_uav_x_forces - uav_x_thrust
    uav_y_dw_forces = smoothed_uav_y_forces - uav_y_thrust
    uav_z_dw_forces = smoothed_uav_z_forces - uav_z_thrust



    # Calculate x-data based on time step
    x = np.arange(len(uav_z_forces)) * dt
    

    # Create figure and subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Plot curves in the first subplot
    
    ax1.plot(x, uav_z_forces, label=f'UAV z-forces')
    ax1.plot(x, uav_z_thrust, label=f'thrust z-forces')
    ax1.plot(x, uav_z_dw_forces, label=f'z-downwash force')
    ax1.plot(x, predictor_forces_z, label=f'predicted z-force')
    ax1.plot(x, [force[2] - 9.81*3.3035 for force in nfc_forces], label=f'NFC z-force')



    #ax1.plot(x, uav_y_forces, label=f'UAV y-forces')
    #ax1.plot(x, uav_z_forces, label=f'UAV z-forces')

    ax1.set_title('First Subplot with Multiple Curves')
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Y-axis Label 1')
    ax1.legend()
    ax1.grid(True)

    # Plot curves in the second subplot
   
    ax2.plot(x, uav_x_dw_forces, label=f'x-force downwash')
    ax2.plot(x, uav_y_dw_forces, label=f'y-force downwash')

    ax2.set_title('Second Subplot with Multiple Curves')
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Y-axis Label 2')
    ax2.legend()
    ax2.grid(True)

    # Adjust layout and show plot
    plt.tight_layout()
    plt.show()

def get_thrust_xyz_forces(uav_state):
    """
    Returns thrust vector of uav as forces in xyz-axes of world
    """
    avg_rps_u2 = np.mean(np.abs(uav_state[22:26]))
    yaw_pitch_roll = uav_state[9:12]
    u2_rot = R.from_euler('xyz', [yaw_pitch_roll[2], yaw_pitch_roll[1], yaw_pitch_roll[0]], degrees=False)
        
    uav_xyz_forces = u2_rot.apply([0, 0, rps_to_thrust_p005_mrv80(avg_rps_u2)])
    
    return uav_xyz_forces


def plot_uav_positions_and_errors(uav1_positions, uav2_positions, target_positions1,target_positions2, dt):
    """
    Plot the xyz positions of two UAVs on multiple subplots, position vs. time, and error vs. time.
    
    Parameters:
    - uav1_positions: List of (x, y, z) positions for UAV 1.
    - uav2_positions: List of (x, y, z) positions for UAV 2.
    - target_positions: List of (x, y, z) target positions.
    - dt: Time step in seconds.
    """
    uav1_positions = np.array(uav1_positions)
    uav2_positions = np.array(uav2_positions)
    target_positions = np.array(target_positions2)
    
    time = np.arange(len(uav1_positions)) * dt
    
    uav1_errors = uav1_positions - target_positions1
    uav2_errors = uav2_positions - target_positions2

    # Create figure and subplots for position plots
    fig, axs = plt.subplots(3, 3, figsize=(15, 12))
    
    # XY, YZ, XZ plots for UAV 1 and UAV 2
    axs[0, 0].plot(uav1_positions[:, 0], uav1_positions[:, 1], label='UAV 1')
    axs[0, 0].plot(uav2_positions[:, 0], uav2_positions[:, 1], label='UAV 2')
    axs[0, 0].set_title('XY Position')
    axs[0, 0].set_xlabel('X Position')
    axs[0, 0].set_ylabel('Y Position')
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    axs[0, 1].plot(uav1_positions[:, 1], uav1_positions[:, 2], label='UAV 1')
    axs[0, 1].plot(uav2_positions[:, 1], uav2_positions[:, 2], label='UAV 2')
    axs[0, 1].set_title('YZ Position')
    axs[0, 1].set_xlabel('Y Position')
    axs[0, 1].set_ylabel('Z Position')
    axs[0, 1].legend()
    axs[0, 1].grid(True)

    axs[0, 2].plot(uav1_positions[:, 0], uav1_positions[:, 2], label='UAV 1')
    axs[0, 2].plot(uav2_positions[:, 0], uav2_positions[:, 2], label='UAV 2')
    axs[0, 2].set_title('XZ Position')
    axs[0, 2].set_xlabel('X Position')
    axs[0, 2].set_ylabel('Z Position')
    axs[0, 2].legend()
    axs[0, 2].grid(True)

    # X, Y, Z positions and target positions over time
    axs[1, 0].plot(time, uav1_positions[:, 0], label='UAV 1 X')
    axs[1, 0].plot(time, uav2_positions[:, 0], label='UAV 2 X')
    axs[1, 0].plot(time, target_positions[:, 0], label='Target X', linestyle='--')
    axs[1, 0].set_title('X Position Over Time')
    axs[1, 0].set_xlabel('Time (seconds)')
    axs[1, 0].set_ylabel('X Position')
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    axs[1, 1].plot(time, uav1_positions[:, 1], label='UAV 1 Y')
    axs[1, 1].plot(time, uav2_positions[:, 1], label='UAV 2 Y')
    axs[1, 1].plot(time, target_positions[:, 1], label='Target Y', linestyle='--')
    axs[1, 1].set_title('Y Position Over Time')
    axs[1, 1].set_xlabel('Time (seconds)')
    axs[1, 1].set_ylabel('Y Position')
    axs[1, 1].legend()
    axs[1, 1].grid(True)

    axs[1, 2].plot(time, uav1_positions[:, 2], label='UAV 1 Z')
    axs[1, 2].plot(time, uav2_positions[:, 2], label='UAV 2 Z')
    axs[1, 2].plot(time, target_positions[:, 2], label='Target Z', linestyle='--')
    axs[1, 2].set_title('Z Position Over Time')
    axs[1, 2].set_xlabel('Time (seconds)')
    axs[1, 2].set_ylabel('Z Position')
    axs[1, 2].legend()
    axs[1, 2].grid(True)

    # Errors over time for X, Y, Z
    axs[2, 0].plot(time, uav1_errors[:, 0], label='UAV 1 X Error')
    axs[2, 0].plot(time, uav2_errors[:, 0], label='UAV 2 X Error')
    axs[2, 0].plot(time, uav1_errors[:, 1], label='UAV 1 Y Error')
    axs[2, 0].plot(time, uav2_errors[:, 1], label='UAV 2 Y Error')
    axs[2, 0].plot(time, uav1_errors[:, 2], label='UAV 1 Z Error')
    axs[2, 0].plot(time, uav2_errors[:, 2], label='UAV 2 Z Error')
    axs[2, 0].set_title('Position Errors Over Time')
    axs[2, 0].set_xlabel('Time (seconds)')
    axs[2, 0].set_ylabel('Error')
    axs[2, 0].legend()
    axs[2, 0].grid(True)

    # Hide unused subplots
    axs[2, 1].axis('off')
    axs[2, 2].axis('off')

    plt.tight_layout()
    plt.show()


# helper functions



# methods for equivariant transformation

def equivariant_agile_transform(u1_states, u2_states, inference=False):
    """
    Returns equivariant representation w.r.t Z-axis
    """

    # extract xy and z from pos, vel, thrust, and 2nd uav's total force f
    dp_xy = u1_states[:,:2] - u2_states[:,:2]
    dp_z = u1_states[:,2] - u2_states[:,2]

    v1_xy = u1_states[:,3:5]
    v1_z = u1_states[:,5]
    v2_xy = u2_states[:,3:5]
    v2_z = u2_states[:,5]

    #v1_xy, _, v2_xy, _ = process_vectors(v1_xy, dp_xy, v2_xy)
    angle_v1_dp_xy = compute_signed_angles(dp_xy, v1_xy)
    angle_v1_v2_xy = compute_signed_angles(v2_xy, v1_xy)


    # compute T1 thrust vectors and their orientations and translations 
    T1 = compute_thrust_vector(u1_states)
    T1_xy = T1[:,:2]
    T1_z = T1[:,2]
    
    T2 = compute_thrust_vector(u2_states)
    #print("T2", T2[:5])
    T2_xy = T2[:,:2]
    T2_z = T2[:,2]

    # symmetry assumptions: flip T1 to left side of dp, and flip T2 accordingly to T2
    #T1_xy_flipped, _, T2_xy, flipped_idx = process_vectors(T1_xy, dp_xy, T2_xy)

    angle_T1_T2_xy = compute_signed_angles(T2_xy, T1_xy)#_flipped)
    angle_T1_dp_xy = compute_signed_angles(dp_xy, T1_xy)#_flipped)

    # force vector f of uav 2
    if not inference:
        f_x, f_y, f_z = extract_uav_forces(u2_states)
        dw_x, dw_y, dw_z = extract_dw_forces(u2_states)
        f_x = dw_x
        f_y = dw_y
    else:
        # during runtime do not compute xyz-residual forces
        #print("inference time....")
        f_x, f_y, f_z = np.zeros((len(dp_xy),1)), np.zeros((len(dp_xy),1)), np.zeros((len(dp_xy),1))

    f_xy = np.column_stack((f_x,f_y))
    #_, _, f_xy, _ = process_vectors(T1_xy, dp_xy, f_xy)

    dw_xy = f_xy


    #print("equiv_agil.dw_xy", dw_xy[:3])
    angle_dw_dp_xy = compute_signed_angles(dp_xy, dw_xy)


    result = (np.linalg.norm(dp_xy, axis=1), dp_z, 
                angle_T1_dp_xy, 
                #np.linalg.norm(T1_xy_flipped, axis=1), T1_z, 
                np.linalg.norm(T1_xy, axis=1), T1_z, 
                np.linalg.norm(T2_xy, axis=1), T2_z, angle_T1_T2_xy, 
                angle_v1_dp_xy, 
                np.linalg.norm(v1_xy, axis=1), v1_z, 
                np.linalg.norm(v2_xy, axis=1), v2_z, angle_v1_v2_xy,
                np.linalg.norm(dw_xy, axis=1), f_z, angle_dw_dp_xy
            )

    # returns |[dp]xy|, [dp]z, 
    # angle_xy(T1,dp), 
    # |[T1]xy|, [T1]z, 
    # |[T2]xy|, [T2]z, angle_xy(T2,T1), 
    # angle_xy(v1,dp), 
    # |[v1]xy|, [v1]z, 
    # |[v2]xy|, [v2]z, angle_xy(v1,v2)
    # [f]xy, [f]z, angle_xy(f, dp)
    return np.column_stack(result), None #flipped_idx


def continous_transform(equivariant_states):
    """
    Input is columns of |dp_xy|, dp_z, angle_T1_dp, |T1_xy|, T1_z, ...
    """
    T1_decomp = angle_demcomposition(equivariant_states[:,2], equivariant_states[:,3])
    T2_decomp = angle_demcomposition(equivariant_states[:,7], equivariant_states[:,5])
    v1_decomp = angle_demcomposition(equivariant_states[:,8], equivariant_states[:,9])
    v2_decomp = angle_demcomposition(equivariant_states[:,13], equivariant_states[:,11])

    dw_decomp = angle_demcomposition(equivariant_states[:,16], equivariant_states[:,14])

    #print("continous_transf.dw_decomp", dw_decomp[:3])

    #dw_decomp = f_decomp - T2_decomp



    result = (
        # z-axis components and dp length
        equivariant_states[:, 0], # |p_xy|
        equivariant_states[:, 1], # p_z
        equivariant_states[:,4], # T1_z
        equivariant_states[:,6], # T2_z
        equivariant_states[:,10], # v1_z
        equivariant_states[:,12], # v2_z
        # new angle independant sinusoidal encodings
        T1_decomp[:,0], T1_decomp[:,1],
        T2_decomp[:,0], T2_decomp[:,1],
        v1_decomp[:,0], v1_decomp[:,1],
        v2_decomp[:,0], v2_decomp[:,1],
        dw_decomp[:,0], dw_decomp[:,1],
    )

    return np.column_stack(result)


def angle_demcomposition(angles, amplitudes):
    # returns sinusoidal encoding of vector a and angle: sin(b)a, cos(b)a
    result = np.column_stack((amplitudes * np.cos(angles), amplitudes * np.sin(angles)))
    return result


def mirror_vectors(v1, v2):
    """Mirror vectors v1 around vectors v2."""
    # Mirror the vectors v1 around v2 using the formula
    dot_product = np.sum(v1 * v2, axis=1)
    norm_v2_squared = np.sum(v2 * v2, axis=1)
    return 2 * (dot_product / norm_v2_squared).reshape(-1, 1) * v2 - v1


def process_vectors(vectors1, vectors2, vectors3):
    """
    If first vector i is to the left of the second vector, mirror it to the right 
    and mirror the vector i of the third list to the opposite side.
    """
    # if T1 is left of dp, mirror T1 to right, and flip T2
    cross_prod = np.cross(vectors1, vectors2)
    mask = cross_prod > 0.0  # means vector1 is to the left of vector2 (positive cross product)
    
    # Mirror T1 and T2 based on mask
    vectors1[mask] = mirror_vectors(vectors1[mask], vectors2[mask])
    vectors3[mask] = mirror_vectors(vectors3[mask], vectors2[mask])

    # where T1 is collinear to dp, bring T2 to right side
    mask_2 = cross_prod == 0.0
    cross_prod32 = np.cross(vectors3, vectors2)
    mask_3 = cross_prod32 > 0.0 # means T2 is to the left of vector2 (positive cross product)
    v1_colin_and_v3_left = mask_2 & mask_3
    vectors3[v1_colin_and_v3_left] = mirror_vectors(vectors3[v1_colin_and_v3_left], 
                                                   vectors2[v1_colin_and_v3_left])


    # summarize where lower vector vector 3, has been mirrored
    mask3 = mask | v1_colin_and_v3_left
    return vectors1, vectors2, vectors3, mask


# triginometric

def compute_signed_angles(v1_list, v2_list):
    """
    Computes the signed and normalized angles between corresponding vectors in two lists.
    
    Parameters:
        v1_list (np.ndarray): Array of shape (N, 2), list of 2D vectors.
        v2_list (np.ndarray): Array of shape (N, 2), list of 2D vectors.
    
    Returns:
        tuple: Two arrays of shape (N,) - signed angles and normalized angles.
    """
    # Ensure inputs are numpy arrays
    v1_list = np.array(v1_list)
    v2_list = np.array(v2_list)
    
    # Compute the signed angle using atan2
    angles = np.arctan2(v2_list[:, 1], v2_list[:, 0]) - np.arctan2(v1_list[:, 1], v1_list[:, 0])
    
    # Normalize angles to [0, 2π)
    angles_normalized = angles % (2 * np.pi)
    
    return angles_normalized
    


def euler_angles_to_quaternion(roll, pitch, yaw):
    """
    Converts Euler angles (attitude angles) to a quaternion.
    
    Parameters:
    roll -- Roll angle [-pi,pi] (in degrees)
    pitch -- Pitch angle [-pi/2, pi/2] (in degrees)
    yaw -- Yaw angle (in degrees)
    
    Returns:
    (qw, qx, qy, qz) -- Corresponding quaternion
    """
    # Convert to raidans
    roll *= np.pi/180
    pitch *= np.pi/180
    yaw *= np.pi/180

    # Calculate half-angles
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    # Calculate quaternion
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return w, x, y, z


def normalize_vector(v, epsilon=1e-12):
    norm = np.linalg.norm(v, axis=-1, keepdims=True)  # Compute norm along the last axis
    
        
    return np.where(norm < epsilon, np.full_like(v, epsilon), v / norm)  # Handle small vectors



def orthogonal_projection(dp_xy, dw_xy_rel):
    # projects dw onto dp as if dp were (1,0)
    e_x = normalize_vector(dp_xy)

    #print("e_x", e_x)
    e_y = np.stack([-e_x[:, 1], e_x[:, 0]], axis=1)
    #print("e_y", e_y)

    #print("dw_xy_rel",dw_xy_rel)

    #print("orthogonal_prog.dw_xy_rel", dw_xy_rel)
    #print("orthogonal_proj.dp",dp_xy)

    dw_abs = dw_xy_rel[:,0:1] * e_x + dw_xy_rel[:,1:2] * e_y

    #print("orthogonal_proj.dw_abs", dw_abs)
    
    return dw_abs





# T = [[0.0,1.0], [-1.0,1.0]

# dp = [[0.0,3.0],[0.0,3.0]]

# signed_angles = compute_signed_angles(dp, T)

# print("angles", (180.0/np.pi) * signed_angles)

# print("magnitudes",np.linalg.norm(T, axis=1))
# decomp = angle_demcomposition(signed_angles, np.linalg.norm(T, axis=1))

# print("decomp", decomp)



# #magn = normalize_vector(T)
# #print("Normalized", magn)
# print("proj", orthogonal_projection(dp, decomp))




class NormalizedMSELoss(torch.nn.Module):

    def __init__(self, sigmas=[1.0, 1.0, 15.0]):
        super().__init__()
        self.sigmas = torch.tensor(sigmas)

    def forward(self, pred, target):
        normalized_error = (pred - target) / self.sigmas.to(pred.device)  
        return torch.mean(normalized_error ** 2)  

class WeightedMSELoss(torch.nn.Module):
    def __init__(self, weight_x=5.0, weight_y=5.0, weight_z=1.0):
        super(WeightedMSELoss, self).__init__()
        self.weight_x = weight_x
        self.weight_y = weight_y
        self.weight_z = weight_z

    def forward(self, predictions, targets):
        pred_x, pred_y, pred_z = predictions[:, 0], predictions[:, 1], predictions[:, 2]
        target_x, target_y, target_z = targets[:, 0], targets[:, 1], targets[:, 2]

        
        mse_x = torch.mean((pred_x - target_x) ** 2)
        mse_y = torch.mean((pred_y - target_y) ** 2)
        mse_z = torch.mean((pred_z - target_z) ** 2)

        
        weighted_mse = self.weight_x * mse_x + self.weight_y * mse_y + self.weight_z * mse_z

        return weighted_mse