# run 2 quadcopter in a flyby scenario and compare all baseline models on the recorded trajectory

from simcontrol import simcontrol2
import numpy as np
import matplotlib.pyplot as plt
import sys, os
from scipy.spatial.transform import Rotation as R

sys.path.append('../../observers/')
sys.path.append('../../uav/')
sys.path.append('../../utils/')

from SO2.model import ShallowEquivariantPredictor
from ndp.model import DWPredictor
from neuralswarm.model import NeuralSwarmPredictor
from empirical.model import EmpiricalPredictor
from analytical.model import AnalyticalPredictor


from uav import *
from utils import *


port = 25556
SIM_DURATION = 11.5
DRONE_TOTAL_MASS = 3.035 # P600 weight
HOVER_TIME = 4.0
FLY_CIRCULAR = False
freq = 0.02
radius = 3.0
y_velocity = 0.8
z_distance = 0.8

# connect to simulators controller
controller = simcontrol2.Controller("localhost", port)

ext_force_sensors = ["force_sensor_body_z", "force_sensor_rotor_1_z", "force_sensor_rotor_2_z", "force_sensor_rotor_3_z", "force_sensor_rotor_4_z"]
jft_sensors = ["jft_sensor_body", "jft_sensor_imu",  "jft_sensor_rotor1", "jft_sensor_rotor2", "jft_sensor_rotor3", "jft_sensor_rotor4"]
joint_sensors = ["joint_sensor_r1", "joint_sensor_r2", "joint_sensor_r3", "joint_sensor_r4",]


# setup UAVs
uav_1_ext_z_force_sensors = ["uav_1_" + ext_sen_name for ext_sen_name in ext_force_sensors]
uav_2_ext_z_force_sensors = ["uav_2_" + ext_sen_name for ext_sen_name in ext_force_sensors]
uav_1_jft_sensors = ["uav_1_" + ext_sen_name for ext_sen_name in jft_sensors]
uav_2_jft_sensors = ["uav_2_" + ext_sen_name for ext_sen_name in jft_sensors]
uav_1_joint_sensors = ["uav_1_" + joint_sen for joint_sen in joint_sensors]
uav_2_joint_sensors = ["uav_2_" + joint_sen for joint_sen in joint_sensors]

uav_1 = UAV("producer", controller, "controller1", "imu1", uav_1_ext_z_force_sensors, uav_1_jft_sensors, uav_1_joint_sensors)
uav_2 = UAV("sufferer", controller, "controller2", "imu2", uav_2_ext_z_force_sensors, uav_2_jft_sensors, uav_2_joint_sensors)

# set experiment 
nan = float('NaN')

# Start simulation
controller.clear()
controller.start()
time_step = controller.get_time_step()  # time_step = 0.0001
sim_max_duration = SIM_DURATION # sim seconds
total_sim_steps = sim_max_duration / time_step
control_frequency = 200.0 # Hz
# Calculate time steps between one control period
steps_per_call = int(1.0 / control_frequency / time_step)
print("One Timestep is ", time_step, "||", "Steps per call are", steps_per_call,"||", 
      "Sim dt is timestep * steps per call =", time_step * steps_per_call)



# Initialize timer
curr_sim_time = 0.0
curr_step = 0

time_seq = []
rel_state_vector_list = []


# each iteration sends control signal
while curr_sim_time < sim_max_duration:

    # log everything at current timestep  first
    time_seq.append(curr_sim_time)
    print("## Sim time:", np.round(curr_sim_time,3), "/", sim_max_duration, "s",
          " ## Sim steps:", curr_step ,"/", total_sim_steps, "steps")
    

    
    if curr_sim_time < HOVER_TIME:
        #px4_input_1 = (0.0, nan,nan,nan, 0.0, 0.0, 0.0, nan, nan, nan, 0.0, nan) 
        #px4_input_1 = (0.0, 0.0, -1.3, 1.5, nan, nan, nan, nan, nan, nan, 1.570796326794897, nan) 
        px4_input_1 = (0.0, 0.0, 1.8, 1.1, nan, nan, nan, nan, nan, nan, -1.570796326794897, nan) 

    else:
        # keep set all acc to 0
        #px4_input_2 = (0.0, 0.0, 1.0, -1.5, nan, nan, nan, nan, nan, nan, 1.570796326794897, nan) 
        
        #px4_input_2 = (0.0, nan, nan, nan, 0.0, 1.5, 0.0, nan, nan, nan, 1.570796326794897, nan) 
        px4_input_1 = (0.0, nan, nan, nan, 0.0, -1.6, 0.0, nan, nan, nan, -1.570796326794897, nan) 


        #px4_input_1 = (0.0, nan,nan,nan, 0.0, 0.0, 0.0, nan, nan, nan, 0.0, nan) 
            
    # second drone just hovering
    #px4_input_1 = (0.0, nan,nan,nan, 0.0, 0.0, 0.0, nan, nan, nan, 0.0, nan) 
    px4_input_2 = (0.0, 0.0, 0.0, 0.0, nan, nan, nan, nan, nan, nan, np.pi, nan)

    # Simulate a control period, giving actuator input and retrieving sensor output
    reply = controller.simulate(steps_per_call,  {
                                                   uav_1.px4_idx: px4_input_1, 
                                                   uav_2.px4_idx: px4_input_2, 
                                                  })
    
    

    # Check simulation result
    if reply.has_error():
        print('Simulation failed! Terminating control experiement.')
        break

    
    # read sensors of uavs
    uav_1.update(reply, curr_sim_time)
    uav_2.update(reply, curr_sim_time)
    
    rel_state_vector_uav_2 = np.array(uav_1.states[-1]) - np.array(uav_2.states[-1])
    
    rel_state_vector_list.append(rel_state_vector_uav_2[:6])
    

    # advance timer and step counter:
    curr_sim_time += steps_per_call * time_step
    curr_step += steps_per_call


# Clear and close simulator
print("Finished, clearing...")
controller.clear()
controller.close()
print("Control experiment ended.")

print("Collected ", len(rel_state_vector_list), "samples of data.")





# Plot recorded and predicted forces
start_idx = 0

uav_1.states = uav_1.states[start_idx:]
uav_2.states = uav_2.states[start_idx:]


mass = 3.035
g = -9.85
mg = np.array([0,0,mass*g])

# 1 load experiment and extract uav 1 and 2 states:

u2_rotations = [R.from_euler('xyz', [yaw_pitch_roll[2], yaw_pitch_roll[1], yaw_pitch_roll[0]],
                               degrees=False) for yaw_pitch_roll in np.array(uav_2.states)[:,9:12]]







# ------------------------



# # Create Rotation objects
# rotations = u2_rotations
# rotations = rotations[0:-1:20]
# positions = np.array(uav_2.states[0:-1:20])[:,0:3]

# # Original vector
# vector = np.array([0, 0, 1])

# # Rotate the vector using the rotations
# rotated_vectors = [rotation.apply(vector) for rotation in rotations]

# # Set up the plot
# fig = plt.figure(figsize=(8, 8))
# ax = fig.add_subplot(111, projection='3d')


# # Plot the rotated vectors
# for i, rotated_vector in enumerate(rotated_vectors):
#     ax.quiver(positions[i][0], positions[i][1], positions[i][2], rotated_vector[0], rotated_vector[1], rotated_vector[2])

# # Set labels and aspect
# ax.set_xlim([-1, 1])
# ax.set_ylim([-1, 1])
# ax.set_zlim([-1, 1])
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# ax.set_title('UAV orientation during flight under downwash')
# ax.legend()

# plt.show()








# ----------------------







u2_avg_rps = np.mean(np.abs(np.array(uav_2.states)[:,19:23]), axis=1)
u2_rps_rot = zip(u2_avg_rps, u2_rotations)

# Find indices where the uavs overlap
y_diff = np.abs(np.array(uav_1.states)[:,1] - np.array(uav_2.states)[:,1])
indices = np.where(y_diff < 0.6)[0]

# find max value of rel. speed
y_max = np.max(np.abs(np.array(uav_1.states)[:,4] - np.array(uav_2.states)[:,4]))



# 2 compute uav actual forces, smooth them, and compute controller z-axis forces, and their residual disturbance
u2_accelerations = np.array(uav_2.states)[:,8]
u2_z_forces = u2_accelerations * mass
smoothed_u2_z_forces = smooth_with_savgol(u2_z_forces, window_size=61, poly_order=1)


Ct =  0.000362
rho = 1.225
A = 0.11948
k = Ct * rho * A
d = 0.3

u2_thrusts = [u2_rotations.apply([0, 0, rps_to_thrust_p005_mrv80(avg_rps)])[2] + (g*mass) for (avg_rps, u2_rotations) in u2_rps_rot]

u2_rps_rot_2 = zip(u2_avg_rps, u2_rotations)
u2_thrusts_formula = [u2_rotations.apply([0, 0, 4.0*k * avg_rps**2.0])[2] + (g*mass) for (avg_rps, u2_rotations) in u2_rps_rot_2]








u2_z_dw_forces = smoothed_u2_z_forces - u2_thrusts





#g_bias_steady_state = 4.5 # on average, Fu compentates always more by 5N, so it is not disturbance force

#dw_forces = np.array(Fz_total) #+ np.array(Fg) - np.array(Fz_u_total)
#dw_forces += g_bias_steady_state
#print("dw forces raw:", dw_forces[-500:-400])
#print("dw forces after subtracting bias:", dw_forces[-500:-400])




time_seq = np.array(time_seq[start_idx:])
rel_state_vector_list = rel_state_vector_list[start_idx:]


plt.vlines(time_seq[[indices[0], indices[-1]]], ymin=min(u2_z_dw_forces), 
           ymax=np.max(u2_thrusts), 
           color='green', linestyle='--', alpha=0.7, label='UAVs overlap region')

#plt.plot(time_seq, u2_z_forces, label='IMU measured z-force of UAV', alpha=0.8)
#plt.plot(time_seq, smoothed_u2_z_forces, label='Smoothed measured z-force of UAV', alpha=0.8)
#plt.plot(time_seq, u2_thrusts, label='control input z-force', alpha=0.5)
#plt.plot(time_seq, u2_thrusts_formula, label='control input z-force (formula)', alpha=0.5)



plt.plot(time_seq, u2_z_dw_forces, label="Downwash residual", color="magenta", linewidth=2)

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

#model = DWPredictor()
#model.load_state_dict(torch.load(model_paths[1], weights_only=True))
#models.append(model)


model = ShallowEquivariantPredictor()
model.load_state_dict(torch.load(model_paths[1], weights_only=True))
models.append(model)

model = ShallowEquivariantPredictor()
model.load_state_dict(torch.load(model_paths[2], weights_only=True))
models.append(model)

#model = NeuralSwarmPredictor()
#model.load_state_dict(torch.load(model_paths[4], weights_only=True))
#models.append(model)

model = EmpiricalPredictor()
models.append(model)
model = AnalyticalPredictor()
models.append(model)


predictions = evaluate_zy_force_curvature(models, np.array(rel_state_vector_list))
labels = [ 
    #"NDP new data no SN",
    "NDP new data SN<4 ", 
    "SO2-Equiv.", 
    "SO2-Equiv. SN<4", 

    # "Nrl.Swarm 2 UAV", 
    "Emprical", "Analytical"
]






for idx, prediction in enumerate(predictions):
    print("plotting pred.")
    plt.plot(time_seq, prediction[:,2], label=labels[idx])


plt.xlabel("time (s)")
plt.ylabel("Force (N)")
plt.title(f"Measured and predicted DW forces for\nTop UAV flies by hovering bottom UAV\nat max. Speed difference of Vy={np.round(y_max,2)}")

plt.grid()
plt.legend()
plt.show()



#np.savez("flyby_below_80_005_vy1_5_rel_state_vector_fu.npz", 
#         time_seq=time_seq, 
#         rel_state_vector_list=rel_state_vector_list,
#         smoothed_u2_z_forces=smoothed_u2_z_forces,
#         u2_z_dw_forces=u2_z_dw_forces, 
#         u2_thrusts=u2_thrusts)



