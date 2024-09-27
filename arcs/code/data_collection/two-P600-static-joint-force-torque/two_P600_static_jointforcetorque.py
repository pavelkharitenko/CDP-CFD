from simcontrol import simcontrol2
import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.append('../../observers/baseline/')
from utils import *



# connect to simulators controller
port = 25556
controller = simcontrol2.Controller("localhost", port)

# retrieve px4 actuator index:
px4_index_1 = controller.get_actuator_info('controller1').index


# setup sensors
imu1_index = controller.get_sensor_info('imu1').index
imu2_index = controller.get_sensor_info('imu2').index

x_force_sensor_idx = controller.get_sensor_info("uav_2_force_sensor_x").index
y_force_sensor_idx = controller.get_sensor_info("uav_2_force_sensor_y").index
z_force_sensor_idx = controller.get_sensor_info("uav_2_force_sensor_z").index

# measure z on the rotors as well:
z_rotor_1_force_sensor_idx = controller.get_sensor_info("uav_2_force_sensor_rotor_1_z").index
z_rotor_2_force_sensor_idx = controller.get_sensor_info("uav_2_force_sensor_rotor_2_z").index
z_rotor_3_force_sensor_idx = controller.get_sensor_info("uav_2_force_sensor_rotor_3_z").index
z_rotor_4_force_sensor_idx = controller.get_sensor_info("uav_2_force_sensor_rotor_4_z").index


uav_2_joint_force_torque_idx = controller.get_sensor_info("uav_2_jft_sensor").index


# set experiment 
nan = float('NaN')

# Start simulation
controller.start()
time_step = controller.get_time_step()  # time_step = 0.0001
sim_max_duration = 1.0 # sim seconds
total_sim_steps = sim_max_duration / time_step
control_frequency = 30.0 # Hz
# Calculate time steps between one control period
steps_per_call = int(1.0 / control_frequency / time_step)
print(
    "One Timestep is ", time_step, "||",
    "Steps per call are", steps_per_call,"||", 
    "Sim dt is timestep * steps per call =", time_step * steps_per_call)


# Initialize timer
curr_sim_time = 0.0
curr_step = 0

time_seq = []
uav_1_pos_list = []
uav_2_pos_list = []

uav_2_external_forces_list = []  # list of (x,y,z) tuples 
rel_state_vector_list = []

dw_joint_sensor_readings = []
dw_body_sensor_readings = []

# each iteration sends control signal
while curr_sim_time < sim_max_duration:

    # log everything at current timestep  first
    time_seq.append(curr_sim_time)
    print("## Sim time:", curr_sim_time, "/", sim_max_duration, "s",
          " ## Sim steps:", curr_step ,"/", total_sim_steps, "steps")
    
    # Create controller input. This is the actuator input vector

    # uav 2 flies to right after 2.6 seconds
    if curr_sim_time < 5.0:
        px4_input_1 = (0.0, 0.0, 1.5, 1.0, nan, nan, nan, nan, nan, nan, 0.0, nan) 
        #px4_input_2 = (0.0, 0.0, -1.5, 1.5, nan, nan, nan, nan, nan, nan, 0.0, nan) 
        pass
    else:
        px4_input_1 = (0.0, 0.0,-1.5, 1.0, nan, nan, nan, nan, nan, nan, 0.0, nan) 
        pass


    # Simulate a control period, giving actuator input and retrieving sensor output
    reply = controller.simulate(steps_per_call,  {
                                                   px4_index_1: px4_input_1, 
                                                  #px4_index_2: px4_input_2
                                                  })

    # Check simulation result
    if reply.has_error():
        print('Simulation failed! Terminating control experiement.')
        break

    
    # Get imu sensor output (a tuple of floats)
    imu1_data = reply.get_sensor_output(imu1_index)
    imu2_data = reply.get_sensor_output(imu2_index)

    uav_2_force_x = reply.get_sensor_output(x_force_sensor_idx)[0]
    uav_2_force_y = reply.get_sensor_output(y_force_sensor_idx)[0]
    uav_2_force_z = reply.get_sensor_output(z_force_sensor_idx)[0]

    uav_2_r1_force_z = reply.get_sensor_output(z_rotor_1_force_sensor_idx)[0]
    uav_2_r2_force_z = reply.get_sensor_output(z_rotor_2_force_sensor_idx)[0]
    uav_2_r3_force_z = reply.get_sensor_output(z_rotor_3_force_sensor_idx)[0]
    uav_2_r4_force_z = reply.get_sensor_output(z_rotor_4_force_sensor_idx)[0]


    uav_2_body_force = reply.get_sensor_output(uav_2_joint_force_torque_idx)

    rounded_body_force = np.round(uav_2_body_force[:3],2)
    print("Force on UAV 2 fixed joint:", rounded_body_force)
    
    uav_2_external_force = np.round((uav_2_force_x, uav_2_force_y, uav_2_force_z),2)
    print("External forces on UAV2: ", uav_2_external_force)

    uav_2_rotor_forces_sum = np.sum(np.round((uav_2_r1_force_z, uav_2_r2_force_z, 
                                                uav_2_r3_force_z, uav_2_r4_force_z),2))
    
    print("Additional rotor z forces: ", uav_2_rotor_forces_sum)

    # evaluate both methods:

    dw_joint_sensor = uav_2_body_force[2] - 29.77
    dw_joint_sensor_readings.append(-dw_joint_sensor)
    dw_body_sensor = np.sum([uav_2_force_z,uav_2_r1_force_z, uav_2_r2_force_z,
                             uav_2_r3_force_z, uav_2_r4_force_z])
    dw_body_sensor_readings.append(dw_body_sensor)


    #print("UAV 2 imu reading:", imu1_data)

    # add x,y,z positions of uav 1
    uav_1_pos = np.array([imu1_data[0], imu1_data[1], imu1_data[2]])
    uav_1_pos_list.append(uav_1_pos)
    # add x,y,z positions of uav 2
    uav_2_pos = np.array([imu2_data[0], imu2_data[1], imu2_data[2]])
    uav_2_pos_list.append(uav_2_pos)
    # add z force of uav 2
    uav_2_external_forces_list.append(uav_2_external_force)
    rel_state_vector = np.round(uav_1_pos_list[-1] - uav_2_pos_list[-1],2)
    print(rel_state_vector)
    rel_state_vector_list.append(rel_state_vector)


    # advance timer and step counter:
    curr_sim_time += steps_per_call * time_step
    curr_step += steps_per_call


# Clear simulator
print("Finished, clearing...")
controller.clear()

# Close the simulation controller
controller.close()
print("Control experiment ended.")

print("Collected ", len(rel_state_vector_list), "samples of data")

plot_3d_vectorfield(rel_state_vector_list, 
                    uav_2_external_forces_list,
                    1/np.max(np.abs(uav_2_external_forces_list)),
                    "\n Collected forces")


fig, axes = plt.subplots(3, 3)
color1 = 'tab:red'
color2 = 'tab:brown'
color3 = 'tab:cyan'
color4 = 'tab:pink'
color5 = 'tab:olive'
color6 = 'tab:purple'


ax1 = axes[0][0]
ax1.set_title('Sufferer UAV Position vs. Time')
ax1.set_xlabel('time (s)')
ax1.set_ylabel('Position (m)', color=color1)
ax1.plot(time_seq, [xyz[2] for xyz in uav_2_pos_list], label='z-axis position', color=color1)
#ax1.plot(seq_t, seq_pos_des[0], label='pos_xd', color=color2)
ax1.tick_params(axis='y', labelcolor=color1)
ax1.legend()

ax2 = axes[1][0]
ax2.set_title('Sufferer UAV x-axis forces vs. Time')
ax2.set_xlabel('time (s)')
ax2.set_ylabel('Force (N)')
ax2.plot(time_seq, [xyz[0] for xyz in uav_2_external_forces_list], label='force_x', color=color2)
#ax1.plot(seq_t, seq_pos_des[0], label='pos_xd', color=color2)
ax2.tick_params(axis='x', labelcolor=color2)
ax2.legend()

ax3 = axes[1][1]
ax3.set_title('Sufferer UAV y-axis forces vs. Time')
ax3.set_xlabel('time (s)')
ax3.set_ylabel('Force (N)')
ax3.plot(time_seq, [xyz[1] for xyz in uav_2_external_forces_list], label='force_y', color=color3)
#ax3.plot(seq_t, seq_pos_des[0], label='pos_xd', color=color2)
ax3.tick_params(axis='y', labelcolor=color3)
ax3.legend()

ax4 = axes[1][2]
ax4.set_title('Sufferer UAV z-axis forces vs. Time')
ax4.set_xlabel('time (s)')
ax4.set_ylabel('Force (N)')
ax4.plot(time_seq, [xyz[2] for xyz in uav_2_external_forces_list], label='force_z', color=color4)
#ax1.plot(seq_t, seq_pos_des[0], label='pos_xd', color=color2)
ax4.tick_params(axis='y')
ax4.legend()

ax4 = axes[2][0]
ax4.set_title('ExternalForceSensor vs JointForceTorqueSensor')
ax4.set_xlabel('time (s)')
ax4.set_ylabel('Force (N)')
ax4.plot(time_seq, dw_body_sensor_readings, label='ExtForSen', color=color4)
ax4.plot(time_seq, dw_joint_sensor_readings, label='JointForTor', color=color5)
ax4.tick_params(axis='y')
ax4.legend()




plt.show()