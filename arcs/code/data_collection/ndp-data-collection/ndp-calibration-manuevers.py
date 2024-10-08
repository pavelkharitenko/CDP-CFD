from simcontrol import simcontrol2
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import sys, time, randomname, pickle
sys.path.append('../../observers/baseline/')
sys.path.append('../../uav/')
sys.path.append('../../utils/')

sys.path.append('../../../../../notify/')
from notify_script_end import notify_ending
from uav import *
from utils import *

SAVE_EXP = True

port = 25556
SIM_DURATION = 577.0
DRONE_TOTAL_MASS = 3.035 # P600 weight
HOVER_TIME = 30.0
FLY_CIRCULAR = False
freq = 0.02
radius = 3.0
nan = float('NaN')

exp_name = init_experiment("Dataset-NDP-2-P600")

#exp = load_forces_from_dataset("2024-10-03-20-13-43-ndp-2-P600-excited-wagon-570.0sec-57001-ts.p")
#extract_labeled_dataset_ndp(exp['uav_list'])
#exit(0)

EXP_SUCCESSFUL = True  # is set to false when errors occur

# connect to simulators controller
controller = simcontrol2.Controller("localhost", port)

ext_force_sensors = ["force_sensor_body_z", "force_sensor_rotor_1_z", "force_sensor_rotor_2_z", "force_sensor_rotor_3_z", "force_sensor_rotor_4_z"]
jft_sensors = ["jft_sensor_body", "jft_sensor_imu",  "jft_sensor_rotor1", "jft_sensor_rotor2", "jft_sensor_rotor3", "jft_sensor_rotor4"]

# setup UAVs
uav_1_ext_z_force_sensors = ["uav_1_" + ext_sen_name for ext_sen_name in ext_force_sensors]
uav_2_ext_z_force_sensors = ["uav_2_" + ext_sen_name for ext_sen_name in ext_force_sensors]
uav_1_jft_sensors = ["uav_1_" + ext_sen_name for ext_sen_name in jft_sensors]
uav_2_jft_sensors = ["uav_2_" + ext_sen_name for ext_sen_name in jft_sensors]

uav_1 = UAV("producer", controller, "controller1", "imu1", uav_1_ext_z_force_sensors, uav_1_jft_sensors)
uav_2 = UAV("sufferer", controller, "controller2", "imu2", uav_2_ext_z_force_sensors, uav_2_jft_sensors)

uav_1_dest = (0,-1.5,4)
uav_2_dest = (0,-1.5,3)

# Start simulation
controller.start()
time_step = controller.get_time_step()  # time_step = 0.0001
sim_max_duration = SIM_DURATION # sim seconds
total_sim_steps = sim_max_duration / time_step
control_frequency = 50.0 # Hz
# Calculate time steps between one control period
steps_per_call = int(1.0 / control_frequency / time_step)
print("One Timestep is ", time_step, "||", "Steps per call are", steps_per_call,"||", 
      "Sim dt is timestep * steps per call =", time_step * steps_per_call)


# Initialize timer
curr_sim_time = 0.0
curr_step = 0
time_seq = []
rel_state_vector_list = []

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
    
    # Create controller input. This is the actuator input vector
    if FLY_CIRCULAR:
        # uav 2 flies to right after 2.6 seconds
        if curr_sim_time < HOVER_TIME:
            px4_input_1 = (0.0, 0.0, 1.5, 1.0, nan, nan, nan, nan, nan, nan, 0.0, nan) 

            pass
        else:
            px = radius * np.sin(2.0 * np.pi * freq * curr_sim_time)
            py = radius - radius * np.cos(2.0 * np.pi * freq * curr_sim_time)
            nan = float('NaN')
            px4_input_1 = (0.0, px, py-1.5, 1.0, nan, nan, nan, nan, nan, nan, 0.0, nan) # This is the actuator input vector

    else:
        if curr_sim_time >= 7.0 and np.round(curr_sim_time,3) % 5.0 == 0.0:
            uav_1_dest = plan_next_coords(0.9, 0.25, uav_1.states[-1][:3], uav_2_dest)
            print("##### Setting new UAV 1 dest", uav_1_dest)

        px4_input_1 = waypoint_after([0.0,4,7],[(0,1.5,4),(0,-1.5,4), uav_1_dest])
        px4_input_2 = waypoint_after([0.0],[uav_2_dest])


        if np.round(curr_sim_time,3) % 50.0 == 0.0:
            if SAVE_EXP:
                uav_1.controller, uav_2.controller = None, None
                exp_path = save_experiment(exp_name, [uav_1, uav_2], EXP_SUCCESSFUL, "intermediate-saving")
                notify_ending("Intermediate saving exp. " + str(exp_path) + " at time " + str(curr_sim_time))
                print("intermediate saved", exp_name)
                uav_1.controller, uav_2.controller = controller, controller
            
    
        
            


    # Simulate a control period, giving actuator input and retrieving sensor output
    reply = controller.simulate(steps_per_call,  {
                                                   uav_1.px4_idx: px4_input_1, 
                                                   uav_2.px4_idx: px4_input_2, 
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


