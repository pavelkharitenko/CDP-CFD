import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../../utils/')
from utils import *

sys.path.append('../../observers/')

from ndp.model import DWPredictor
from SO2.model import ShallowEquivariantPredictor
from agile.model import AgileLeanPredictor, AgileEquivariantPredictor
#from model import DWPredictor


roll_iterations = False

dataset_paths = [
        #r"C:\Users\admin\Desktop\IDP\CDP-CFD\arcs\code\data_collection\agile_manuevers\1_flybelow\raw_data_1_flybelow_200Hz_80_005_len34951ts_51_iterations_testset.npz",
        #r"C:\Users\admin\Desktop\IDP\CDP-CFD\arcs\code\data_collection\agile_manuevers\2_flyabove\raw_data_2_flyabove_200Hz_80_005_len32956ts_46_iterations_testset.npz",
        r"C:\Users\admin\Desktop\IDP\CDP-CFD\arcs\code\data_collection\agile_manuevers\3_swapping\raw_data_3_swapping_200Hz_80_005_len29736ts_52_iterations_testset.npz"
]



uav_1_states = []
uav_2_states = []
time_seq = []

for path in dataset_paths:
    
    data = np.load(path)
    uav_1_state, uav_2_state = data['uav_1_states'], data['uav_2_states']

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


overlap_indices = np.where(np.abs(np.array(uav_1_states)[:,1] - np.array(uav_2_states)[:,1]) < 0.6)[0]
overlaps = np.array(time_seq)[overlap_indices]

# 2 compute uav actual forces, smooth them, and compute controller z-axis forces, and their residual disturbance
u2_accelerations = np.array(uav_2_states)[:,8]
u2_z_forces = u2_accelerations * mass
smoothed_u2_z_forces = smooth_with_savgol(u2_z_forces, window_size=21, poly_order=1)
u2_thrusts = [u2_rotations.apply([0, 0, rps_to_thrust_p005_mrv80(avg_rps)])[2] + (g*mass) for (avg_rps, u2_rotations) in u2_rps_rot]
u2_z_dw_forces = smoothed_u2_z_forces - u2_thrusts


model_paths = [

r"C:\Users\admin\Desktop\IDP\CDP-CFD\arcs\code\observers\ndp\2024-12-04-20-20-39-NDP-sn-None-123-wintry-polygon20000_eps.pth",
r"C:\Users\admin\Desktop\IDP\CDP-CFD\arcs\code\observers\ndp\2024-12-04-20-28-52-NDP-sn-4-123-presto-vulture20000_eps.pth",
#r"C:\Users\admin\Desktop\IDP\CDP-CFD\arcs\code\observers\SO2\2024-11-20-00-10-30-SO2-Model-below-sn_scale-None-gray-javelin20000_eps.pth",
#r"C:\Users\admin\Desktop\IDP\CDP-CFD\arcs\code\observers\SO2\2024-11-20-12-30-17-SO2-Model-below-sn_scale-4-dull-flow20000_eps.pth",
#r"C:\Users\admin\Desktop\IDP\CDP-CFD\arcs\code\observers\agile\2024-12-04-17-01-01-Agile-sn-None-123-cyclic-swamp30000_eps.pth",
#r"C:\Users\admin\Desktop\IDP\CDP-CFD\arcs\code\observers\agile\2024-12-04-22-16-05-Agile-sn-None-123-braised-upload50000_eps.pth",
r"C:\Users\admin\Desktop\IDP\CDP-CFD\arcs\code\observers\agile\2024-12-04-23-05-47-Agile-Equiv-sn-None-123-cloying-glass20000_eps.pth",
#r"C:\Users\admin\Desktop\IDP\CDP-CFD\arcs\code\observers\agile\2024-12-04-22-33-58-Agile-lessdata-sn-None-123-syrupy-pitch60000_eps.pth",
#r"C:\Users\admin\Desktop\IDP\CDP-CFD\arcs\code\observers\agile\2024-12-04-20-43-28-Agile-sn-None-123-cheerful-tint20000_eps.pth",
r"C:\Users\admin\Desktop\IDP\CDP-CFD\arcs\code\observers\agile\2024-12-04-20-48-02-Agile-sn-4-123-recursive-gallon20000_eps.pth",
#r"C:\Users\admin\Desktop\IDP\CDP-CFD\arcs\code\observers\agile\2024-12-04-21-29-04-Agile-sn-4-123-plain-channel6800_eps.pth",
#r"C:\Users\admin\Desktop\IDP\CDP-CFD\arcs\code\observers\agile\2024-12-04-22-02-16-Agile-sn-3-123-woolen-run10000_eps.pth",
]
models = []

colors= ["green", "purple", "grey", "red", "cyan", "pink", "black"]



model = DWPredictor()
model.load_state_dict(torch.load(model_paths[0], weights_only=True))
models.append(model)

model = DWPredictor()
model.load_state_dict(torch.load(model_paths[1], weights_only=True))
models.append(model)

#model = ShallowEquivariantPredictor()
#model.load_state_dict(torch.load(model_paths[2], weights_only=True))
#models.append(model)

model_al = AgileEquivariantPredictor()
model_al.load_state_dict(torch.load(model_paths[2], weights_only=True))

model_al2 = AgileLeanPredictor()
model_al2.load_state_dict(torch.load(model_paths[3], weights_only=True))


predictions = evaluate_zy_force_curvature(models, np.array(rel_state_vector_list)[:,:6])
predictions.append(model_al.evaluate(np.array(uav_1_states), np.array(uav_2_states)))
predictions.append(model_al2.evaluate(np.array(uav_1_states), np.array(uav_2_states)))



labels = [ 
"NDP no SN",
"NDP SN<4 ", 
#"SO2-Equiv.", 
#"SO2-Equiv. SN<4", 
"Agile Equiv.",
"Agile Lean SN<3",
#"Agile SN<4 - 6000 epochs"
]


fig = plt.subplot()
#fig.plot(u2_z_forces, label="UAV's z-axis forces") # unsmoothed
#plot_array_with_segments(fig, time_seq, smoothed_u2_z_forces, color="blue", roll=roll_iterations, label="UAV total Z-forces")
#plot_array_with_segments(fig, time_seq, u2_thrusts,  color="orange", roll=roll_iterations, label="controller z-forces")
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
# find indices where the time array decreases (reset points) for number of iterations
reset_indices = np.where(np.diff(time_seq) < 0)[0] + 1  # Add 1 to shift to the start of the next segment
reset_indices = np.append([0], reset_indices)  # Include the start of the array as the first segment boundary
reset_indices = np.append(reset_indices, len(time_seq))  # Include the end of the array as the last boundary
n_itrs = len(reset_indices)