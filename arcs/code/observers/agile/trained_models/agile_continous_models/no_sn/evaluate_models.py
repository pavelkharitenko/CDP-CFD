import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../../utils/')
from utils import *
from sklearn.metrics import r2_score

sys.path.append('../../observers/')

from neuralswarm.model import NeuralSwarmPredictor
from ndp.model import DWPredictor
from SO2.model import ShallowEquivariantPredictor
from agile.model import AgileContinousPredictor, AgileShallowPredictor
from NNARX.model import NNARXModel
#from model import DWPredictor


roll_iterations = False

dataset_paths = [
        r"C:\Users\admin\Desktop\IDP\CDP-CFD\arcs\code\data_collection\agile_manuevers\1_flybelow\speeds_05_20\raw_data_1_flybelow_200Hz_80_005_len34951ts_51_iterations_testset.npz",
        r"C:\Users\admin\Desktop\IDP\CDP-CFD\arcs\code\data_collection\agile_manuevers\2_flyabove\speeds_05_20\raw_data_2_flyabove_200Hz_80_005_len32956ts_46_iterations_testset.npz", 
        r"C:\Users\admin\Desktop\IDP\CDP-CFD\arcs\code\data_collection\agile_manuevers\3_swapping\speeds_05_20\raw_data_3_swapping_200Hz_80_005_len29736ts_52_iterations_testset.npz",
        r"C:\Users\admin\Desktop\IDP\CDP-CFD\arcs\code\data_collection\agile_manuevers\3_swapping\speeds_20_40\raw_data_3_swapping_fast_200Hz_80_005_len20802ts_67_iterations_testset.npz",
        
]

dataset_titles = [
    "1 fly below",
    "2 fly above",
    "3 swapping"
]



uav_1_states = []
uav_2_states = []
time_seq = []

for path in dataset_paths:
    
    data = np.load(path)
    uav_1_state, uav_2_state = data['uav_1_states'][:-1], data['uav_2_states'][:-1]

    uav_1_states.extend(uav_1_state)
    uav_2_states.extend(uav_2_state)
    time_seq.extend(np.arange(0,len(uav_1_state)))



plot = True
mass = 3.035
g = -9.85


# 1 extract necessary parameters from uav states
rel_state_vector_list = np.array(uav_1_states) - np.array(uav_2_states)
u2_rotations = [R.from_euler('xyz', [yaw_pitch_roll[2], yaw_pitch_roll[1], yaw_pitch_roll[0]], degrees=False) for yaw_pitch_roll in np.array(uav_2_states)[:,9:12]]
u2_avg_rps = np.mean(np.abs(np.array(uav_2_states)[:,22:26]), axis=1)
u2_rps_rot = zip(u2_avg_rps, u2_rotations)


overlap_indices = np.where(np.abs(np.array(uav_1_states)[:,1] - np.array(uav_2_states)[:,1]) < 10.5)[0]
overlaps = np.array(time_seq)[overlap_indices]

# 2 compute uav actual forces, smooth them, and compute controller z-axis forces, and their residual disturbance
u2_accelerations = np.array(uav_2_states)[:,8]
u2_z_forces = u2_accelerations * mass
smoothed_u2_z_forces = smooth_with_savgol(u2_z_forces, window_size=21, poly_order=1)
u2_thrusts = [u2_rotations.apply([0, 0, rps_to_thrust_p005_mrv80(avg_rps)])[2] + (g*mass) for (avg_rps, u2_rotations) in u2_rps_rot]
u2_z_dw_forces = smoothed_u2_z_forces - u2_thrusts


model_paths = [
#r"C:\Users\admin\Desktop\IDP\CDP-CFD\arcs\code\observers\neuralswarm\2024-12-09-23-07-36-NSwarm-sn-None-123S-tall-hearth10000_eps.pth",
#r"C:\Users\admin\Desktop\IDP\CDP-CFD\arcs\code\observers\neuralswarm\2024-12-09-23-07-36-NSwarm-sn-None-123S-tall-hearth20000_eps.pth",
#r"C:\Users\admin\Desktop\IDP\CDP-CFD\arcs\code\observers\neuralswarm\2024-12-09-23-07-36-NSwarm-sn-None-123S-tall-hearth50000_eps.pth",
r"C:\Users\admin\Desktop\IDP\CDP-CFD\arcs\code\observers\agile\trained_models\agile_continous_models\123S\2024-12-09-15-49-54-Agile-Cont-full-data-sn-None-123S-broad-ramp10000_eps.pth",
# NNARX
#r"C:\Users\admin\Desktop\IDP\CDP-CFD\arcs\code\observers\NNARX\2024-12-10-16-19-51-NNARX-lags-1-123S-factorial-radiator20_eps.pth",
#r"C:\Users\admin\Desktop\IDP\CDP-CFD\arcs\code\observers\NNARX\2024-12-10-16-37-41-NNARX-lags-5-123S-chamfered-artifact20_eps.pth",
#r"C:\Users\admin\Desktop\IDP\CDP-CFD\arcs\code\observers\NNARX\2024-12-10-16-43-39-NNARX-lags-10-123S-modern-buyer20_eps.pth",
r"C:\Users\admin\Desktop\IDP\CDP-CFD\arcs\code\observers\NNARX\2024-12-10-16-47-28-NNARX-lags-15-123S-warped-vocoder20_eps.pth",
r"C:\Users\admin\Desktop\IDP\CDP-CFD\arcs\code\observers\agile\2024-12-11-19-55-09-Agile-Shllw-full-data-sn-None-123S-future-country30000_eps.pth",
r"C:\Users\admin\Desktop\IDP\CDP-CFD\arcs\code\observers\agile\2024-12-11-19-55-09-Agile-Shllw-full-data-sn-None-123S-future-country20000_eps.pth",
r"C:\Users\admin\Desktop\IDP\CDP-CFD\arcs\code\observers\agile\2024-12-11-19-55-09-Agile-Shllw-full-data-sn-None-123S-future-country10000_eps.pth"]
models = []

colors= ["green", "purple", "grey", "red", "cyan", "pink", "black"]


# model = NeuralSwarmPredictor()
# model.load_state_dict(torch.load(model_paths[0], weights_only=True))
# models.append(model)

# model = NeuralSwarmPredictor()
# model.load_state_dict(torch.load(model_paths[1], weights_only=True))
# models.append(model)


model_al1 = AgileContinousPredictor()
model_al1.load_state_dict(torch.load(model_paths[0], weights_only=True))

model_al2 = NNARXModel(input_dim=210)
model_al2.load_state_dict(torch.load(model_paths[1], weights_only=True))

model_al3 = AgileShallowPredictor()
model_al3.load_state_dict(torch.load(model_paths[2], weights_only=True))

model_al4 = AgileShallowPredictor()
model_al4.load_state_dict(torch.load(model_paths[3], weights_only=True))

model_al5 = AgileShallowPredictor()
model_al5.load_state_dict(torch.load(model_paths[4], weights_only=True))

#model = ShallowEquivariantPredictor()
#model.load_state_dict(torch.load(model_paths[1], weights_only=True))
#models.append(model)


# model_al1 = AgileContinousPredictor()
# model_al1.load_state_dict(torch.load(model_paths[1], weights_only=True))

# model_al2 = AgileContinousPredictor()
# model_al2.load_state_dict(torch.load(model_paths[2], weights_only=True))

# model_al3 = AgileContinousPredictor()
# model_al3.load_state_dict(torch.load(model_paths[3], weights_only=True))






predictions = evaluate_zy_force_curvature(models, np.array(rel_state_vector_list)[:,:6])
predictions.append(model_al1.evaluate(np.array(uav_1_states), np.array(uav_2_states)))
predictions.append(model_al2.evaluate(np.array(uav_1_states), np.array(uav_2_states)))
predictions.append(model_al3.evaluate(np.array(uav_1_states), np.array(uav_2_states)))
predictions.append(model_al4.evaluate(np.array(uav_1_states), np.array(uav_2_states)))
predictions.append(model_al5.evaluate(np.array(uav_1_states), np.array(uav_2_states)))





labels = [ 
"Agile 20k full data",
"NNARX 15",
"shllw 10",
"shllw 20",
"Shllw 30",
]




fig = plt.subplot()
#fig.plot(u2_z_forces, label="UAV's z-axis forces") # unsmoothed
#plot_array_with_segments(fig, time_seq, smoothed_u2_z_forces, color="blue", roll=roll_iterations, label="UAV total Z-forces")
#plot_array_with_segments(fig, time_seq, u2_thrusts,  color="orange", roll=roll_iterations, label="controller z-forces")
plot_array_with_segments(fig, np.array(time_seq)[overlaps], u2_z_dw_forces[overlaps], color="magenta", roll=roll_iterations, label="downwash disturbance forces", overlaps=overlaps)

# evaluate predictors
for idx, prediction in enumerate(predictions):
    #scenario_title = dataset_titles[idx]
    print("Now:", labels[idx])

    


    
    plot_array_with_segments(fig, np.array(time_seq)[overlaps], prediction[:,2][overlaps], roll=roll_iterations, color=colors[idx], label=labels[idx])
    compute_rmse(prediction[:,2][overlaps], u2_z_dw_forces[overlaps],label=labels[idx])
    print(f"R2 score of {labels[idx]}", r2_score(u2_z_dw_forces[overlaps],prediction[:,2][overlaps]))



    




plt.ylabel("Force [N]")
plt.xlabel("time [s]")
plt.grid()
plt.title("Actual bottom UAV Z-forces, controller's thrust, and residual downwash force")
plt.legend()
plt.show()



# 4 create label:
u2_z_dw_forces = np.array([[0,0,z_force] for z_force in u2_z_dw_forces])

# label format v1: [state_uav1, state_uav2, dw_forces]
# find indices where the time array decreases (reset points) for number of iterations
reset_indices = np.where(np.diff(time_seq) < 0)[0] + 1  # Add 1 to shift to the start of the next segment
reset_indices = np.append([0], reset_indices)  # Include the start of the array as the first segment boundary
reset_indices = np.append(reset_indices, len(time_seq))  # Include the end of the array as the last boundary
n_itrs = len(reset_indices)