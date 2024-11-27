import torch, pickle, randomname, sys, math
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from numpy import random as rnd
from shapely.geometry import Polygon, Point
sys.path.append('../../uav/')
from uav import *



from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation as R

sys.path.append('../../../observers/')

from ndp.model import DWPredictor
from SO2.model import ShallowEquivariantPredictor


device = "cuda" if torch.cuda.is_available() else "cpu"

def plot_xy_slices(model):
    """
    Visualize (x,y,z) output of NN as 2D heatmap. Based on "plot_historgrams" at
    https://github.com/Li-Jinjie/ndp_nmpc_qd/blob/master/ndp_nmpc/scripts/dnwash_nn_est/nn_train.py
    """
    
    # Plot xy-slices at different Z-values:
    xy_plane_z_samples = [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5]
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

def plot_zy_yx_slices(model):
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


def plot_z_slices(model):
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


def plot_so2_xy_slice(model):
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


def plot_so2_zy_xy_slices(model):
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


def euler_angles_to_quaternion(roll, pitch, yaw):
    """
    Converts Euler angles (attitude angles) to a quaternion.
    
    Parameters:
    roll -- Roll angle [-pi,pi] (in degrees)
    pitch -- Pitch angle [-pi/2, pi/2] (in degrees)
    yaw -- Yaw angle (in degrees)
    
    Returns:
    (qx, qy, qz, qw) -- Corresponding quaternion
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
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy

    return qx, qy, qz, qw




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


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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

        for path in dataset_paths:
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

    
    overlap_indices = np.where(np.abs(np.array(uav_1_states)[:,1] - np.array(uav_2_states)[:,1]) < 0.6)[0]
    overlaps = np.array(time_seq)[overlap_indices]

    # 2 compute uav actual forces, smooth them, and compute controller z-axis forces, and their residual disturbance
    u2_accelerations = np.array(uav_2_states)[:,8]
    u2_z_forces = u2_accelerations * mass
    smoothed_u2_z_forces = smooth_with_savgol(u2_z_forces, window_size=21, poly_order=1)
    u2_thrusts = [u2_rotations.apply([0, 0, rps_to_thrust_p005_mrv80(avg_rps)])[2] + (g*mass) for (avg_rps, u2_rotations) in u2_rps_rot]
    u2_z_dw_forces = smoothed_u2_z_forces - u2_thrusts

    
    model_paths = [
        r"C:\Users\admin\Desktop\IDP\CDP-CFD\arcs\code\observers\ndp\2024-11-20-13-11-05-NDP-predictor-sn_scale-4-300k-ts-flyby-navy-sill20000_eps.pth",
        r"C:\Users\admin\Desktop\IDP\CDP-CFD\arcs\code\observers\SO2\2024-11-20-00-10-30-SO2-Model-below-sn_scale-None-gray-javelin20000_eps.pth",
        r"C:\Users\admin\Desktop\IDP\CDP-CFD\arcs\code\observers\SO2\2024-11-20-12-30-17-SO2-Model-below-sn_scale-4-dull-flow20000_eps.pth",
    ]
    models = []

    colors= ["green", "yellow", "red"]

    model = DWPredictor()
    model.load_state_dict(torch.load(model_paths[0], weights_only=True))
    models.append(model)

    model = ShallowEquivariantPredictor()
    model.load_state_dict(torch.load(model_paths[1], weights_only=True))
    models.append(model)

    model = ShallowEquivariantPredictor()
    model.load_state_dict(torch.load(model_paths[2], weights_only=True))
    models.append(model)


    predictions = evaluate_zy_force_curvature(models, np.array(rel_state_vector_list)[:,:6])
    labels = [ 
        "NDP new data SN<4 ", 
        "SO2-Equiv.", 
        "SO2-Equiv. SN<4", 
    ]



    # 3 plot if needed
    if plot:
        fig = plt.subplot()
        #fig.plot(u2_z_forces, label="UAV's z-axis forces") # unsmoothed
        plot_array_with_segments(fig, time_seq, smoothed_u2_z_forces, color="blue", roll=roll_iterations, label="controller z-forces")
        plot_array_with_segments(fig, time_seq, u2_thrusts,  color="orange", roll=roll_iterations, label="controller z-forces")
        plot_array_with_segments(fig, time_seq, u2_z_dw_forces, color="magenta", roll=roll_iterations, label="downwash disturbance forces", overlaps=overlaps)
        
        # evaluate predictors
        for idx, prediction in enumerate(predictions):
            plot_array_with_segments(fig, time_seq, prediction[:,2], roll=roll_iterations, color=colors[idx], label=labels[idx])
            compute_rmse(prediction[:,2][overlap_indices], u2_z_dw_forces[overlap_indices],label=labels[idx])
            
        
        



        plt.ylabel("Force [N]")
        plt.xlabel("time [s]")
        plt.grid()
        plt.title("Actual bottom UAV Z-forces, controller's thrust, and residual downwash force")
        plt.legend()
        plt.show()



    # 4 create label:
    u2_z_dw_forces = np.array([[0,0,z_force] for z_force in u2_z_dw_forces])

    # label format v1: [state_uav1, state_uav2, dw_forces]
    if save:
        np.savez(f"raw_data_1_flybelow_200Hz_80_005_len{len(uav_1_states)}ts", 
                 uav_1_states=uav_1_states, 
                 uav_2_states=uav_2_states, 
                 dw_forces=u2_z_dw_forces)




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