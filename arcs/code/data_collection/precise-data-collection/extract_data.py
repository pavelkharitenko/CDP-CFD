import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
sys.path.append('../../utils/')
from utils import *

dataset_paths = [
    r"C:\Users\admin\Desktop\IDP\CDP-CFD\arcs\code\data_collection\precise-data-collection\2024-11-18-15-45-47-Precise-collection-descriptive-strategyintermediate-120.0sec-48001-ts.p",
    r"C:\Users\admin\Desktop\IDP\CDP-CFD\arcs\code\data_collection\precise-data-collection\2024-11-18-16-38-22-Precise-collection-legacy-treeintermediate-300.0sec-120001-ts.p",
    r"C:\Users\admin\Desktop\IDP\CDP-CFD\arcs\code\data_collection\precise-data-collection\2024-11-18-18-33-13-Precise-collection-humble-filterintermediate-240.0sec-96001-ts.p",
    r"C:\Users\admin\Desktop\IDP\CDP-CFD\arcs\code\data_collection\precise-data-collection\2024-11-18-19-07-55-Precise-collection-bone-bookintermediate-120.0sec-48001-ts.p"
]

uav_1_states = []
uav_2_states = []


for path in dataset_paths:
    exp = load_forces_from_dataset(path)
    uav_1, uav_2 = exp['uav_list']
    uav_1_states.extend(uav_1.states[3000:0])
    uav_2_states.extend(uav_2.states[3000:0])

plot = False
mass = 3.035
g = -9.85
mg = np.array([0,0,mass*g])

# 1 load experiment and extract uav 1 and 2 states:

u2_rotations = [R.from_euler('xyz', [yaw_pitch_roll[2], yaw_pitch_roll[1], yaw_pitch_roll[0]], degrees=False) for yaw_pitch_roll in np.array(uav_2_states)[:,9:12]]
u2_avg_rps = np.mean(np.abs(np.array(uav_2_states)[:,16:20]), axis=1)
u2_rps_rot = zip(u2_avg_rps, u2_rotations)

overlap = [i for i, (a, b) in enumerate(zip(np.array(uav_1_states)[:,1], np.array(uav_2_states)[:,1])) if abs(a - b) < 0.6]

# 2 compute uav actual forces, smooth them, and compute controller z-axis forces, and their residual disturbance
u2_accelerations = np.array(uav_2_states)[:,8]
u2_z_forces = u2_accelerations * mass
smoothed_u2_z_forces = smooth_with_savgol(u2_z_forces, window_size=61, poly_order=1)
u2_thrusts = [u2_rotations.apply([0, 0, rps_to_thrust_p005_mrv80(avg_rps)])[2] + (g*mass) for (avg_rps, u2_rotations) in u2_rps_rot]

u2_z_dw_forces = smoothed_u2_z_forces - u2_thrusts

# 3 plot if needed
if plot:
    fig = plt.subplot()
    #fig.plot(u2_z_forces, label="UAV's z-axis forces")


    #fig.plot(moving_average(u2_z_forces, 20), label="UAV's z-axis forces smoothed (running avg)")
    fig.vlines(overlap, ymin=min(u2_z_dw_forces), ymax=max(u2_thrusts), color='grey', alpha=0.1, label="UAVs closer < 0.6 on y axis")
    fig.plot(smoothed_u2_z_forces, label="UAV's z-axis forces smoothed (sav.gol.)")
    fig.plot(u2_thrusts, label="controller's z-forces")
    fig.plot(u2_z_dw_forces, color="magenta",label="downwash disturbance forces")
    plt.ylabel("Force (N)")
    plt.title("Actual bottom UAV Z-forces, controller's thrust, and residual downwash force")

    plt.legend()
    plt.show()



# 4 create label:



