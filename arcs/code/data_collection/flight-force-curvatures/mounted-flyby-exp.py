from simcontrol import simcontrol2
from datetime import datetime
import keyboard
import numpy as np
import matplotlib.pyplot as plt
import sys, time, randomname, pickle
from numpy import random as rnd

sys.path.append('../../uav/')
sys.path.append('../../utils/')

from uav import *
from utils import *

SAVE_EXP = False

port = 25556
SIM_DURATION = 7.0
DRONE_TOTAL_MASS = 3.035 # P600 weight
nan = float('NaN')

sys.path.append('../../../../../notify/')
from notify_script_end import notify_ending

exp_name = init_experiment("ndp-2-P600-low")


y_velocity = 0.15




EXP_SUCCESSFUL = True  # is set to false when errors occur

# connect to simulators controller
controller = simcontrol2.Controller("localhost", port)

ext_force_sensors = ["force_sensor_body_z", "force_sensor_rotor_1_z", "force_sensor_rotor_2_z", "force_sensor_rotor_3_z", "force_sensor_rotor_4_z"]
jft_sensors = ["jft_sensor_body", "jft_sensor_imu",  "jft_sensor_rotor1", "jft_sensor_rotor2", "jft_sensor_rotor3", "jft_sensor_rotor4"]

# setup uav 2 mounted jft sensor and rotor motors
uav_2_world_joint_idx = controller.get_sensor_info("uav_2_jft_sensor").index
uav_2_r1_idx = controller.get_actuator_info("uav_2_r1_joint_motor").index
uav_2_r2_idx = controller.get_actuator_info("uav_2_r2_joint_motor").index
uav_2_r3_idx = controller.get_actuator_info("uav_2_r3_joint_motor").index
uav_2_r4_idx = controller.get_actuator_info("uav_2_r4_joint_motor").index


# setup UAVs
uav_1_ext_z_force_sensors = ["uav_1_" + ext_sen_name for ext_sen_name in ext_force_sensors]
uav_2_ext_z_force_sensors = ["uav_2_" + ext_sen_name for ext_sen_name in ext_force_sensors]
uav_1_jft_sensors = ["uav_1_" + ext_sen_name for ext_sen_name in jft_sensors]
uav_2_jft_sensors = ["uav_2_" + ext_sen_name for ext_sen_name in jft_sensors]

uav_1 = UAV("producer", controller, "controller1", "imu1", uav_1_ext_z_force_sensors, uav_1_jft_sensors)
uav_2 = UAV("sufferer", controller, "uav_2_r1_joint_motor", "imu2", uav_2_ext_z_force_sensors, uav_2_jft_sensors, "uav_2_jft_sensor")



# Start simulation
controller.start()
time_step = controller.get_time_step()  # time_step = 0.0001
sim_max_duration = SIM_DURATION # sim seconds
total_sim_steps = sim_max_duration / time_step
control_frequency = 30.0 # Hz
# Calculate time steps between one control period
steps_per_call = int(1.0 / control_frequency / time_step)
print("One Timestep is ", time_step, "||", "Steps per call are", steps_per_call,"||", 
      "Sim dt is timestep * steps per call =", time_step * steps_per_call)


# Initialize timer
curr_sim_time = 0.0
curr_step = 0
time_seq = []
rel_state_vector_list = []
mounted_jft_sensor_list = []
radius = 0.5



# each iteration sends control signal
while curr_sim_time < sim_max_duration:

    # log everything at current timestep  first
    time_seq.append(curr_sim_time)
    print("#### Sim time:", np.round(curr_sim_time,3), "/", sim_max_duration, "s",
          " ## Sim steps:", curr_step ,"/", total_sim_steps, "steps")

    px4_input_1 = (0.0, 
                   0, -0.5, 0.5, 
                   nan, nan, nan, 
                    nan, nan, nan, 
                   0.0, nan) 
    
    if curr_sim_time > 1.0:
        px4_input_1 = (0.0, 
                    nan, nan, 1.0, 
                   0.0, y_velocity, nan, 
                    nan, nan, nan, 
                   0.0, nan) 
        

    # Simulate a control period, giving actuator input and retrieving sensor output
    reply = controller.simulate(steps_per_call,  {
                                                   uav_1.px4_idx: px4_input_1, 
                                                   uav_2_r1_idx: (353,), 
                                                   uav_2_r2_idx: (353,), 
                                                   uav_2_r3_idx: (-353,),  # front right
                                                   uav_2_r4_idx: (-353,),  # back left
                                                  })
    
    # Check simulation result
    if reply.has_error():
        print('Simulation failed! Terminating control experiement.')
        notify_ending("script failed at time " + str(curr_sim_time))
        EXP_SUCCESSFUL = False
        break
    
    # read sensors of uavs
    uav_1.update(reply, curr_sim_time)
    uav_2.update(reply, curr_sim_time)

    rel_state_vector_uav_2 = np.round(np.array(uav_1.states[-1]) - np.array(uav_2.states[-1]), 2)
    #print(rel_state_vector)
    rel_state_vector_list.append(rel_state_vector_uav_2)

    uav_2_mounted_jft_sensor =  reply.get_sensor_output(uav_2_world_joint_idx)
    mounted_jft_sensor_list.append(uav_2_mounted_jft_sensor)
    

    
    print("Relative state vector:", rel_state_vector_list[-1][:3])
    print("Recorded force:", np.round(uav_2_mounted_jft_sensor[2], 2))
    print("-----------------------------------------")
    
    
    # advance timer and step counter:
    curr_sim_time += steps_per_call * time_step
    curr_step += steps_per_call


# Clear and close simulator
print("Finished, clearing...")
controller.clear()
controller.close()
print("Control loop ended.")
print("Collected ", len(time_seq), "samples of data.")




rec_dw_forces = [jft_meas[2] for jft_meas in mounted_jft_sensor_list]
#print(rec_dw_forces)

model_paths = [r"C:\Users\admin\Desktop\IDP\CDP-CFD\arcs\code\observers\ndp\2024-10-09-12-25-27-NDP-Li-Model-sn_scale-None-144k-datapoints-corrected-bias-complicated-tropics20000_eps.pth",
               r"C:\Users\admin\Desktop\IDP\CDP-CFD\arcs\code\observers\ndp\2024-10-09-21-24-10-NDP-Li-Model-sn_scale-4-144k-datapoints-corrected-bias-sizzling-speed20000_eps.pth",
               r"C:\Users\admin\Desktop\IDP\CDP-CFD\arcs\code\observers\ndp\2024-10-10-10-56-56-NDP-Li-Model-sn_scale-2-144k-datapoints-corrected-bias-edible-status20000_eps.pth",
               r"C:\Users\admin\Desktop\IDP\CDP-CFD\arcs\code\observers\ndp\2024-10-09-10-25-10-NDP-Li-Model-sn_scale-None-114k-datapoints-inventive-defilade20000_eps.pth",
               r"C:\Users\admin\Desktop\IDP\CDP-CFD\arcs\code\observers\ndp\2024-10-09-10-47-22-NDP-Li-Model-sn_scale-4-114k-datapoints-stubborn-content20000_eps.pth"
               ]

models = []
for model_path in model_paths:
    model = DWPredictor().to(device)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    models.append(model)

predictions = evaluate_zy_force_curvature(models, rel_state_vector_list)
labels = ["Li's NDP (no sn)","sn<=4", "sn <=2", "no sn, old bias","sn<4 old bias"]
plt.plot(time_seq,[-jft_meas[2] for jft_meas in mounted_jft_sensor_list], label='Recorded z-force')

for idx, prediction in enumerate(predictions):
    print("plotting pred.")
    plt.plot(time_seq, prediction, label=labels[idx])


plt.xlabel("time (s)")
plt.ylabel("Force (N)")
plt.title(f"Measured and predicted DW forces at\nSpeed of Y={y_velocity}")
plt.legend()
plt.show()








if SAVE_EXP:
    uav_1.controller, uav_2.controller = None, None
    exp_path = save_experiment(exp_name, [uav_1, uav_2], EXP_SUCCESSFUL, SIM_DURATION)
    notify_ending("DATA COLLECTION FINISHED: saved exp as " + str(exp_path))




plt.show()


