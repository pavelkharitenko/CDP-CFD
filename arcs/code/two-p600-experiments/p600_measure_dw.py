from simcontrol import simcontrol2
import numpy as np
import matplotlib.pyplot as plt


# connect to simulators controller
port = 25556
controller = simcontrol2.Controller("localhost", port)

# retrieve px4 actuator index:
px4_index_1 = controller.get_actuator_info('controller1').index


# setup IMUs
# Retrieve imu sensor index (imu is a class of sensor)
imu1_index = controller.get_sensor_info('imu1').index
imu2_index = controller.get_sensor_info('imu2').index
force_sensor_idx = controller.get_sensor_info("uav_2_force_sensor_z").index

# set experiment 
nan = float('NaN')

# Start simulation
controller.start()
time_step = controller.get_time_step()  # time_step = 0.0001
sim_max_duration = 15.0 # sim seconds
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

uav_2_force_z_list = []

# each iteration sends control signal
while curr_sim_time < sim_max_duration:

    # log everything at current timestep  first
    time_seq.append(curr_sim_time)
    print("## Sim time:", curr_sim_time, "/", sim_max_duration, "s",
          " ## Sim steps:", curr_step ,"/", total_sim_steps, "steps")
    
    # Create controller input. This is the actuator input vector

    # uav 2 flies to right after 2.6 seconds
    if curr_sim_time < 3.0:
        px4_input_1 = (0.0, 0.0, 1.5, 1.0, nan, nan, nan, nan, nan, nan, 0.0, nan) 
        #px4_input_2 = (0.0, 0.0, -1.5, 1.5, nan, nan, nan, nan, nan, nan, 0.0, nan) 
        pass
    else:
        px4_input_1 = (0.0, 0.0,-1.5, 1.0, nan, nan, nan, nan, nan, nan, 0.0, nan) 
        pass

    #px4_input_2 = (0.0, 0.0, -1.5, 1.5, nan, nan, nan, nan, nan, nan, 0.0, nan) 

    # uav 1  flies over left
    #if curr_sim_time < 7.6:
    #    px4_input_1 = (0.0, 0.0, 10.0, 2.0, nan, nan, nan, nan, nan, nan, 0.0, nan) 
    #else:
    #    px4_input_1 = (0.0, 0.0, 10.0, 2.0, nan, nan, nan, nan, nan, nan, 0.0, nan) 

    # Simulate a control period, giving actuator input and retrieving sensor output
    reply = controller.simulate(steps_per_call,  {px4_index_1: px4_input_1, 
                                                  #px4_index_2: px4_input_2
                                                  })


    
    # Get imu sensor output (a tuple of floats)
    imu1_data = reply.get_sensor_output(imu1_index)
    imu2_data = reply.get_sensor_output(imu2_index)
    uav_2_force_z = reply.get_sensor_output(force_sensor_idx)

    print("Z-axis forces on UAV2: ", uav_2_force_z)

    # add x,y,z positions of uav 1
    uav_1_pos_list.append((imu1_data[0], imu1_data[1], imu1_data[2]))

    # add x,y,z positions of uav 2
    uav_2_pos_list.append((imu2_data[0], imu2_data[1], imu2_data[2]))

    # add z force of uav 2
    uav_2_force_z_list.append(uav_2_force_z)


    # advance timer and step counter:
    curr_sim_time += steps_per_call * time_step
    curr_step += steps_per_call


# Clear simulator
print("Finished, clearing...")
controller.clear()

# Close the simulation controller
controller.close()
print("Done.")


fig, axes = plt.subplots(2, 2)
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
ax1.plot(time_seq, [xyz[2] for xyz in uav_2_pos_list], label='pos_z', color=color1)
#ax1.plot(seq_t, seq_pos_des[0], label='pos_xd', color=color2)
ax1.tick_params(axis='y', labelcolor=color1)
ax1.legend()

ax2 = axes[0][1]
ax2.set_title('Sufferer UAV z-axis forces vs. Time')
ax2.set_xlabel('time (s)')
ax2.set_ylabel('Force (N)', color=color1)
ax2.plot(time_seq, [xyz[0] for xyz in uav_2_force_z_list], label='force_z', color=color2)
#ax1.plot(seq_t, seq_pos_des[0], label='pos_xd', color=color2)
ax2.tick_params(axis='y', labelcolor=color1)
ax2.legend()


plt.show()