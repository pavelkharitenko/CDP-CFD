import sys, torch
sys.path.append('../../utils/')
from utils import *

sys.path.append('../../observers/')

from neuralswarm.model import NeuralSwarmPredictor
from ndp.model import DWPredictor
from SO2.model import ShallowEquivariantPredictor
from agile.model import AgileShallowPredictor, AgilePredictor
from NNARX.model import NNARXModel
from analytical.model import AnalyticalPredictor, AnalyticalPredictorVectorized

predictor = AgilePredictor(output_dim=3)

predictor.load_state_dict(torch.load(find_file_with_substring("proper-hardball"), weights_only=True))


import numpy as np

def generate_dummy_uav_states(xy_range, plane_res, z_point):
    """
    Generate dummy UAV states for visualization.
    UAV 2 is fixed at [0, 0, 0], and UAV 1 is positioned around it.
    """
    xy_samples = np.linspace(start=-xy_range, stop=xy_range, num=plane_res)
    uav_1_states = []
    uav_2_states = []

    for i in range(plane_res):
        for j in range(plane_res):
            # UAV 1 position
            uav_1_pos = [xy_samples[i], xy_samples[j], z_point]
            # UAV 2 position (fixed at origin)
            uav_2_pos = [0.0, 0.0, 0.0]

            # Create full state vectors with zeros for velocities, accelerations, etc.
            uav_1_state = uav_1_pos + [0.0] * 21  # 21 zeros for the rest of the state
            uav_2_state = uav_2_pos + [0.0] * 21  # 21 zeros for the rest of the state

            uav_1_states.append(uav_1_state)
            uav_2_states.append(uav_2_state)

    return np.array(uav_1_states), np.array(uav_2_states)




def plot_xy_slices_agile(model):
    """
    Visualize (x,y,z) output of NN as 2D heatmap.
    """
    # Plot xy-slices at different Z-values:
    xy_plane_z_samples = [-0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
    xy_range = 1.0  # size of XY plane
    plane_res = 200  # number of points sampled for plotting in one dimension

    color = "autumn_r"
    fig, ax = plt.subplots(2, len(xy_plane_z_samples), sharex=True, sharey=True, figsize=(6, 6))

    model.eval()
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    plotted_forces = []  # save plotted forces for adjusting min and max value of heatmaps
    all_imgs = []  # save all images for colorbar

    # loop over Z-heights and generate plane_res*plane_res sample points
    for idx, z_point in enumerate(xy_plane_z_samples):
        uav_1_states, uav_2_states = generate_dummy_uav_states(xy_range, plane_res, z_point)

        # Evaluate the model
        dw_forces = model.evaluate(uav_1_states, uav_2_states)
        zs = dw_forces[:, 2]  # Extract z-component of the forces

        # Reshape the forces into a 2D grid for plotting
        plot_f = zs.reshape(plane_res, plane_res)

        # Plot the heatmap
        im = ax[0][idx].imshow(plot_f, extent=[-xy_range, xy_range, xy_range, -xy_range], cmap=color)
        all_imgs.append(im)
        ax[0][idx].set_title(f"Z = {z_point}m")
        plotted_forces.extend(zs)

    # Add colorbar
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)



plot_xy_slices_ndp(predictor)