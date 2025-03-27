import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
sys.path.append('../../utils/')

sys.path.append('../../observers/')

from ndp.model import DWPredictor
from SO2.model import ShallowEquivariantPredictor
from utils import *

dataset_paths = [
    r"C:\Users\admin\Desktop\IDP\CDP-CFD\arcs\code\data_collection\precise-data-collection\2024-11-19-21-23-22-Precise-collection-tart-slalom-800.0sec-115136-ts.p"
]

uav_1_states = []
uav_2_states = []


for path in dataset_paths:
    exp = load_forces_from_dataset(path)
    uav_1, uav_2 = exp['uav_list']
    uav_1_states.extend(uav_1.states[25000:35000])
    uav_2_states.extend(uav_2.states[25000:35000])

plot = True
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



model_paths = [
    r"C:\Users\admin\Desktop\IDP\CDP-CFD\arcs\code\observers\ndp\2024-11-20-13-11-05-NDP-predictor-sn_scale-4-300k-ts-flyby-navy-sill20000_eps.pth",
    #r"C:\Users\admin\Desktop\IDP\CDP-CFD\arcs\code\observers\SO2\2024-11-20-00-10-30-SO2-Model-below-sn_scale-None-gray-javelin10000_eps.pth",
    r"C:\Users\admin\Desktop\IDP\CDP-CFD\arcs\code\observers\SO2\2024-11-20-00-10-30-SO2-Model-below-sn_scale-None-gray-javelin20000_eps.pth",
    r"C:\Users\admin\Desktop\IDP\CDP-CFD\arcs\code\observers\SO2\2024-11-20-12-30-17-SO2-Model-below-sn_scale-4-dull-flow20000_eps.pth",
    #r"C:\Users\admin\Desktop\IDP\CDP-CFD\arcs\code\observers\SO2\2024-11-20-12-50-10-SO2-Model-below-sn_scale-6-pink-basket20000_eps.pth",
    #r"C:\Users\admin\Desktop\IDP\CDP-CFD\arcs\code\observers\SO2\2024-11-20-12-50-10-SO2-Model-below-sn_scale-6-pink-basket40000_eps.pth",
]

models = []


model = DWPredictor()
model.load_state_dict(torch.load(model_paths[0], weights_only=True))
models.append(model)


model = ShallowEquivariantPredictor()
model.load_state_dict(torch.load(model_paths[1], weights_only=True))
models.append(model)

model = ShallowEquivariantPredictor()
model.load_state_dict(torch.load(model_paths[2], weights_only=True))
models.append(model)

#model = ShallowEquivariantPredictor()
#model.load_state_dict(torch.load(model_paths[3], weights_only=True))
#models.append(model)

# model = ShallowEquivariantPredictor()
# model.load_state_dict(torch.load(model_paths[4], weights_only=True))
# models.append(model)



predictions = evaluate_zy_force_curvature(models, np.array(uav_1_states)[:,:6] - np.array(uav_2_states)[:,:6])
labels = [ 
    #"NDP new data no SN",
    "NDP new data SN<4 ",
    #"SO2-Equiv. 40k ep", 
    #"SO2-Equiv. 10k ep", 
    "SO2-Equiv. 20k ep", 
    "SO2-Equiv. 20k ep, SN<4", 
    #"SO2-Equiv. 20k ep, SN<6", 
    #"SO2-Equiv. 40k ep, SN<6", 


    #"SO2-Equiv. 40k ep, SN <4", 



    # "Nrl.Swarm 2 UAV", 
    #"Emprical", "Analytical"
]









# 3 plot if needed
if plot:
    fig = plt.subplot()
    #fig.plot(u2_z_forces, label="UAV's z-axis forces")


    #fig.plot(moving_average(u2_z_forces, 20), label="UAV's z-axis forces smoothed (running avg)")
    fig.vlines(overlap, ymin=min(u2_z_dw_forces), ymax=max(u2_thrusts), color='grey', alpha=0.1, label="UAVs closer < 0.6 on y axis")
    #fig.plot(smoothed_u2_z_forces, label="UAV's z-axis forces smoothed (sav.gol.)")
    #fig.plot(u2_thrusts, label="controller's z-forces")
    fig.plot(u2_z_dw_forces, color="magenta",label="downwash disturbance forces", linewidth=2)
    for idx, prediction in enumerate(predictions):
        print("plotting pred.")
        plt.plot(prediction[:,2], label=labels[idx], linewidth=2)


    plt.ylabel("Force (N)")
    plt.title("Actual bottom UAV Z-forces, controller's thrust, and residual downwash force")
    plt.grid()
    plt.legend()
    plt.show()






# plot for unseen data:

data = np.load(r"C:\Users\admin\Desktop\IDP\CDP-CFD\arcs\code\data_collection\flight-force-curvatures\flyby_below_80_005_vy1_5_rel_state_vector_fu.npz")
time_seq = data["time_seq"]
rel_state_vector_list=data["rel_state_vector_list"]
smoothed_u2_z_forces=data["smoothed_u2_z_forces"]
u2_z_dw_forces=data["u2_z_dw_forces"]
u2_thrusts=data["u2_thrusts"]
start_idx = 0



time_seq = np.array(time_seq[start_idx:])
rel_state_vector_list = rel_state_vector_list[start_idx:]

indices = np.where(rel_state_vector_list[:,1] < 0.3)[0]

plt.vlines(time_seq[[indices[0], indices[-1]]], ymin=min(u2_z_dw_forces), 
           ymax=np.max(u2_thrusts), 
           color='green', linestyle='--', alpha=0.7, label='UAVs overlap region')

#plt.plot(time_seq, u2_z_forces, label='IMU measured z-force of UAV', alpha=0.8)
#plt.plot(time_seq, smoothed_u2_z_forces, label='Smoothed measured z-force of UAV', alpha=0.8)
#plt.plot(time_seq, u2_thrusts, label='control input z-force', alpha=0.5)

plt.plot(time_seq, u2_z_dw_forces, color="magenta", linewidth=2, label="Downwash residual")



predictions = evaluate_zy_force_curvature(models, np.array(rel_state_vector_list))




for idx, prediction in enumerate(predictions):
    print("plotting pred.")
    plt.plot(time_seq, prediction[:,2], label=labels[idx], linewidth=2)




plt.xlabel("time (s)")
plt.ylabel("Force (N)")
#plt.title(f"Measured and predicted DW forces for\nTop UAV flies by hovering bottom UAV\nat max. Speed difference of Vy={np.round(y_max,2)}")

plt.grid()
plt.legend()
plt.show()

