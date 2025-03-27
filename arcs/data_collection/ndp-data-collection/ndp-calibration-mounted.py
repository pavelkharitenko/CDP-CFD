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

SAVE_EXP = True

port = 25556
SIM_DURATION = 600.00
DRONE_TOTAL_MASS = 3.035 # P600 weight


nan = float('NaN')

sys.path.append('../../../../../notify/')
from notify_script_end import notify_ending

exp_name = init_experiment("ndp-2-P600-low")


keys_pressed_tpl = [0,0,0]
vel_gain = 0.05

manual_control = True

def update_tpl(i,val):
    global keys_pressed_tpl

    keys_pressed_tpl[i] += val * vel_gain
    if keys_pressed_tpl[i] > 0.5:
        keys_pressed_tpl[i] = 0.5
    if keys_pressed_tpl[i] < -0.5:
        keys_pressed_tpl[i] = -0.5

def toggle_control():
    global manual_control
    manual_control = not manual_control
    

keyboard.add_hotkey('left', update_tpl, args=[1,-1])
keyboard.add_hotkey('right', update_tpl, args=[1,1])
keyboard.add_hotkey('up', update_tpl, args=[0,-1])
keyboard.add_hotkey('down', update_tpl, args=[0,1])
keyboard.add_hotkey('i', update_tpl, args=[2,1])
keyboard.add_hotkey('k', update_tpl, args=[2,-1])
keyboard.add_hotkey('c', toggle_control)




#exp = load_forces_from_dataset("2024-10-03-20-13-43-ndp-2-P600-excited-wagon-570.0sec-57001-ts.p")
#extract_labeled_dataset_ndp(exp['uav_list'])
#exit(0)

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

uav_1_dest = (0,0,0)
uav_2_dest = (0,0,0)

# Start simulation
controller.start()
time_step = controller.get_time_step()  # time_step = 0.0001
sim_max_duration = SIM_DURATION # sim seconds
total_sim_steps = sim_max_duration / time_step
control_frequency = 100.0 # Hz
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
uav_1_z_pos = -.2
radius = 0.5

def waypoint_after(timepoints, waypoints):
    next_goal = None
    for i, timepoint in enumerate(timepoints):
        if curr_sim_time >= timepoint:
             next_goal = (0.0, waypoints[i][0], waypoints[i][1], waypoints[i][2], nan, nan, nan, nan, nan, nan, 0.0, nan)
    return next_goal


# each iteration sends control signal
while curr_sim_time < sim_max_duration:

    # log everything at current timestep  first
    time_seq.append(curr_sim_time)
    print("#### Sim time:", np.round(curr_sim_time,3), "/", sim_max_duration, "s",
          " ## Sim steps:", curr_step ,"/", total_sim_steps, "steps")
    


    freq = 0.5
    
    px = radius * np.sin(2.0 * np.pi * freq * curr_sim_time)
    py = radius * np.cos(2.0 * np.pi * freq * curr_sim_time) 
    nan = float('NaN')
    px4_input_1 = (0.0, px, py, uav_1_z_pos, nan, nan, nan, nan, nan, nan, 0.0, nan) 
    print(px4_input_1)
    if np.round(curr_sim_time,3) % 3 == 0:
        uav_1_z_pos = rnd.uniform(-0.3, -0.1)
        radius =  rnd.uniform(0.5, 1)


    # save uavs states every n seconds
    if np.round(curr_sim_time,3) % 50.0 == 0.0 and np.round(curr_sim_time,3) >= 1.0:
        if SAVE_EXP:
            uav_1.controller, uav_2.controller = None, None
            exp_path = save_experiment(exp_name, [uav_1, uav_2], EXP_SUCCESSFUL, "intermediate-saving", bias=29.77)
            notify_ending("Intermediate saving exp. " + str(exp_path) + " at time " + str(curr_sim_time))
            print("intermediate saved", exp_name)
            uav_1.controller, uav_2.controller = controller, controller
            
   

    # if manual_control:
    #     px4_input_1 = (0.0, nan, nan, nan,
    #                     keys_pressed_tpl[0], 
    #                     keys_pressed_tpl[1], 
    #                     keys_pressed_tpl[2], 
    #                     nan, nan, nan, 0.0, nan)
        
    #     print("Sent inputs:", np.round(keys_pressed_tpl,2))

    # else:
    #     keys_pressed_tpl = [0,0,0]
    #     uav_1_z_pos = uav_1.states[-1][2]
    #     px = radius * np.sin(2.0 * np.pi * freq * curr_sim_time) 
    #     py = radius - radius * np.cos(2.0 * np.pi * freq * curr_sim_time)
    #     px4_input_1 = (0.0, px, py, uav_1_z_pos, nan, nan, nan, nan, nan, nan, 0.0, nan)
        
            


    # Simulate a control period, giving actuator input and retrieving sensor output
    reply = controller.simulate(steps_per_call,  {
                                                   uav_1.px4_idx: px4_input_1, 
                                                   #uav_2_r1_idx: (353,), 
                                                   #uav_2_r2_idx: (353,), 
                                                   #uav_2_r3_idx: (-353,),  # front right
                                                   #uav_2_r4_idx: (-353,),  # back left
                                                   uav_2_r1_idx: (0,), 
                                                   uav_2_r2_idx: (0,), 
                                                   uav_2_r3_idx: (0,),  # front right
                                                   uav_2_r4_idx: (0,),  # back left
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
    #print("Recorded forces on UAV 2 mount:", )

    print("Collected", len(rel_state_vector_list), "amount of data")
    print("Control Mode:", str(manual_control))
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

# save experiment
#notify_ending("script has ended just now.")

plot_uav_statistics([uav_1, uav_2])

if SAVE_EXP:
    uav_1.controller, uav_2.controller = None, None
    exp_path = save_experiment(exp_name, [uav_1, uav_2], EXP_SUCCESSFUL, SIM_DURATION)
    notify_ending("DATA COLLECTION FINISHED: saved exp as " + str(exp_path))
    plot_uav_statistics([uav_1, uav_2])
    load_forces_from_dataset(exp_path)

#plot_3d_vectorfield(rel_state_vector_list, uav_2_jt_forces_list, 1.0/np.max(np.abs(uav_2_jt_forces_list)), "\n Collected forces")
#plot_uav_force_statistics(uav_1.timestamp_list, uav_1.ext_forces_list, uav_1.jft_forces_list,  [uav_1.mass*state[8] for state in uav_1.states])



plt.show()


