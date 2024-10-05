from simcontrol import simcontrol2
import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.append('../../observers/baseline/')
sys.path.append('../../uav/')
from uav import *
from utils import *

port = 25556
SIM_DURATION = 30.0
DRONE_TOTAL_MASS = 3.035 # P600 weight
HOVER_TIME = 30.0
FLY_CIRCULAR = False
freq = 0.02
radius = 3.0

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

# set experiment 
nan = float('NaN')

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


# each iteration sends control signal
while curr_sim_time < sim_max_duration:

    # log everything at current timestep  first
    time_seq.append(curr_sim_time)
    print("## Sim time:", np.round(curr_sim_time,3), "/", sim_max_duration, "s",
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
        if curr_sim_time < HOVER_TIME:
            #px4_input_1 = (0.0, nan,nan,nan, 0.0, 0.0, 0.0, nan, nan, nan, 0.0, nan) 
            px4_input_1 = (0.0, 0.0, 1.5, 1.0, nan, nan, nan, nan, nan, nan, 0.0, nan) 
        else:
            # keep position
            px4_input_1 = (0.0, 0.0, 1.5, 1.0, nan, nan, nan, nan, nan, nan, 0.0, nan) 
            #px4_input_1 = (0.0, nan,nan,nan, 0.0, 0.0, 0.0, nan, nan, nan, 0.0, nan) 
            
    # second drone just hovering
    px4_input_2 = (0.0, 0.0, -1.5, 1.0, nan, nan, nan, nan, nan, nan, 0.0, nan)

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
print("Control experiment ended.")

print("Collected ", len(rel_state_vector_list), "samples of data.")

#plot_3d_vectorfield(rel_state_vector_list, uav_2_jt_forces_list, 1.0/np.max(np.abs(uav_2_jt_forces_list)), "\n Collected forces")

#plot_uav_force_statistics(uav_1.timestamp_list, uav_1.ext_forces_list, uav_1.jft_forces_list,  [uav_1.mass*state[8] for state in uav_1.states])

plot_uav_statistics([uav_1,uav_2])


plt.show()

exit(0)
plt.axis('tight')
plt.show()

print("----- Evaluation External Forces ------")
print("Mean ext forces on rotor 1:",np.mean(uav_1_external_force_rotor1))
print("Mean ext forces on rotor 2:",np.mean(uav_1_external_force_rotor2))
print("Mean ext forces on rotor 3:",np.mean(uav_1_external_force_rotor3))
print("Mean ext forces on rotor 4:",np.mean(uav_1_external_force_rotor4))
print("Mean ext forces on body:",np.mean(uav_1_external_force_body))
print("Total mean Z external forces:", np.mean([np.sum(xyz) for xyz in zip(uav_1_external_force_body, uav_1_external_force_rotor1, uav_1_external_force_rotor2,
                                              uav_1_external_force_rotor3, uav_1_external_force_rotor4)]))
print("----- Evaluation JointForceTorque at rotors -----")
print("Mean jft force on body:",np.mean([body_r1_r2_r3_r4[0][2] for body_r1_r2_r3_r4 in uav_1_jtf_list]))
print("Mean jft force on imu:",np.mean([body_r1_r2_r3_r4[1][2] for body_r1_r2_r3_r4 in uav_1_jtf_list]))
print("Mean jft force on rotor 1:",np.mean([body_r1_r2_r3_r4[2][2] for body_r1_r2_r3_r4 in uav_1_jtf_list]))
print("Mean jft force on rotor 2:",np.mean([body_r1_r2_r3_r4[3][2] for body_r1_r2_r3_r4 in uav_1_jtf_list]))
print("Mean jft force on rotor 3:",np.mean([body_r1_r2_r3_r4[4][2] for body_r1_r2_r3_r4 in uav_1_jtf_list]))
print("Mean jft force on rotor 4:",np.mean([body_r1_r2_r3_r4[5][2] for body_r1_r2_r3_r4 in uav_1_jtf_list]))
print("Total mean sum Z jft forces:", np.mean([body_r1_r2_r3_r4[2] for body_r1_r2_r3_r4 in np.sum(uav_1_jtf_list, axis=1)]))




