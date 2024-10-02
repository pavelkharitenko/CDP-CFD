from simcontrol import simcontrol2
import numpy as np
import matplotlib.pyplot as plt


SIM_DURATION = 60.0

freq = 0.02
radius = 3.0


# connect to simulators controller
port = 25556
controller = simcontrol2.Controller("localhost", port)

px4_index = controller.get_actuator_info('controller').index
imu_index = controller.get_sensor_info('imu').index

external_z_force_idx = controller.get_sensor_info("ext_force_sensor_rotor_1_z").index
jft_sensor_idx = controller.get_sensor_info("jft_sensor_rotor_1").index
nan = float('NaN')
px4_input = (0.0, 0.0, 0.0, 2.0, nan, nan, nan, nan, nan, nan, 0.0, nan) # This is the actuator input vector


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
print(
    "One Timestep is ", time_step, "||",
    "Steps per call are", steps_per_call,"||", 
    "Sim dt is timestep * steps per call =", time_step * steps_per_call)


# Initialize timer
curr_sim_time = 0.0
curr_step = 0

time_seq = []
external_z_forces_list = []
jft_forces_list = []

# each iteration sends control signal
while curr_sim_time < sim_max_duration:

    # log everything at current timestep  first
    time_seq.append(curr_sim_time)
    print("## Sim time:", curr_sim_time, "/", sim_max_duration, "s",
          " ## Sim steps:", curr_step ,"/", total_sim_steps, "steps")
    
    px4_input = (0.0, 0.0, 0.0, 2.0, nan, nan, nan, nan, nan, nan, 0.0, nan) # This is the actuator input vector
    

    # Simulate a control period, giving actuator input and retrieving sensor output
    reply = controller.simulate(steps_per_call,  {px4_index: px4_input})
    
    print("#### Reply", reply)

    # Check simulation result
    if reply.has_error():
        print('Simulation failed! Terminating control experiement.')
        break

    
    # Get imu sensor output (a tuple of floats)


   


    ext_force = reply.get_sensor_output(external_z_force_idx)
    print("External force reads:", ext_force)
    external_z_forces_list.append(ext_force)


    
    jft_force = reply.get_sensor_output(jft_sensor_idx)
    print("Jft force:", jft_force)
    jft_forces_list.append(jft_force)
   
    
    # advance timer and step counter:
    curr_sim_time += steps_per_call * time_step
    curr_step += steps_per_call


# Clear and close simulator
print("Finished, clearing...")
controller.clear()
controller.close()
print("Control experiment ended.")

exit()

print("Collected ", len(rel_state_vector_list), "samples of data.")

plot_3d_vectorfield(rel_state_vector_list, 
                    uav_2_jt_forces_list,
                    1.0/np.max(np.abs(uav_2_jt_forces_list)),
                    "\n Collected forces")


fig, axes = plt.subplots(3, 3)
color1 = 'tab:red'
color2 = 'tab:brown'
color3 = 'tab:cyan'
color4 = 'tab:pink'
color5 = 'tab:olive'
color6 = 'tab:purple'


# Plot Both drones position
ax1 = axes[0][0]
ax1.set_title('DW Producer & Sufferer UAV XY Positions')
ax1.scatter([xyz[1] for xyz in uav_1_state_list], [xyz[2] for xyz in uav_1_state_list], label='Producer UAV')
ax1.scatter([xyz[1] for xyz in uav_2_state_list], [xyz[2] for xyz in uav_2_state_list], label='Sufferer UAV')
ax1.set_xlabel('Y-Position (m)')
ax1.set_ylabel('Z-Position (m)')
ax1.legend()


# Plot recorded forces over time
ax2 = axes[1][0]
ax2.set_title('External Forces on Sufferer UAV')
ax2.set_xlabel('Y-Position (m) of Producer')
ax2.set_ylabel('Force (N)')
ax2.plot([xyz[1] for xyz in uav_1_state_list], [xyz[0] for xyz in uav_2_jt_forces_list], label='force_x')
ax2.plot([xyz[1] for xyz in uav_1_state_list], [xyz[1] for xyz in uav_2_jt_forces_list], label='force_y')
ax2.plot([xyz[1] for xyz in uav_1_state_list], [xyz[2] for xyz in uav_2_jt_forces_list], label='force_z jft sensor')
ax2.plot([xyz[1] for xyz in uav_1_state_list], dw_body_sensor_readings, label='force_z ef sensor')
ax2.legend()

ax3 = axes[0][1]
ax3.set_title('UAV Y-Positions over time')
ax3.set_xlabel('time (s)')
ax3.set_ylabel('Y-Position (m)')
ax3.plot(time_seq, [xyz[1] for xyz in uav_1_state_list], label='Y-Pos Producer')
ax3.plot(time_seq, [xyz[1] for xyz in uav_2_state_list], label='Y-Pos Sufferer')

#ax3.plot(time_seq, [xyz[1] for xyz in uav_2_jt_forces_list], label='force_y')
ax3.legend()


ax4 = axes[1][1]
ax4.set_title('ExtForce vs JointForTor sensor Z force comparison')
ax4.set_xlabel('time (s)')
ax4.set_ylabel('Force (N)')
ax4.plot(time_seq, dw_body_sensor_readings, label='ExtForSen', color=color4)
ax4.plot(time_seq, dw_joint_sensor_readings, label='JointForTor', color=color5)
ax4.legend()

ax3 = axes[2][1]
ax3.set_title('Sufferer UAV External X & Y-axis forces')
ax3.set_xlabel('time (s)')
ax3.set_ylabel('Force (N)')
ax3.plot(time_seq, [xyz[0] for xyz in uav_2_jt_forces_list], label='force_x')
ax3.plot(time_seq, [xyz[1] for xyz in uav_2_jt_forces_list], label='force_y')
ax3.legend()


ax3 = axes[0][2]
ax3.set_title('Producer UAV External Z force on rotor 4')
ax3.set_xlabel('time (s)')
ax3.set_ylabel('Force (N)')
ax3.plot(time_seq, uav_1_external_force_rotor4, label='r4 force_z')
#ax3.plot(time_seq, [xyz[1] for xyz in uav_2_jt_forces_list], label='force_y')
ax3.legend()


plt.show()

print("Mean forces on rotor 4",np.mean(uav_1_external_force_rotor4))