from simcontrol import simcontrol2
import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.append('../../observers/baseline/')
from utils import *

SIM_DURATION = 60.0



def read_sensor(reply, sensor_idx, indicies=[0]):
    result = []
    sensor_tuple_data = reply.get_sensor_output(sensor_idx)
    for i in indicies:
        result.append(sensor_tuple_data[i])
    if len(result) == 1:
        return result[0]
    else:
        return result


def read_multiple_sensors(reply, sensor_indicies):
    result = []
    for sensor_idx in sensor_indicies:
        result.append(read_sensor(reply, sensor_idx))
    return result

# connect to simulators controller
port = 25556
controller = simcontrol2.Controller("localhost", port)

# retrieve px4 actuator index:
px4_index_1 = controller.get_actuator_info('controller1').index


# setup sensors
imu1_index = controller.get_sensor_info('imu1').index
imu2_index = controller.get_sensor_info('imu2').index

fx_sensor_idx = controller.get_sensor_info("uav_2_force_sensor_x").index
fy_sensor_idx = controller.get_sensor_info("uav_2_force_sensor_y").index
fz_sensor_idx = controller.get_sensor_info("uav_2_force_sensor_z").index

# measure z on the rotors as well:
uav2_r1_z_force_idx = controller.get_sensor_info("uav_2_force_sensor_rotor_1_z").index
uav2_r2_z_force_idx = controller.get_sensor_info("uav_2_force_sensor_rotor_2_z").index
uav2_r3_z_force_idx = controller.get_sensor_info("uav_2_force_sensor_rotor_3_z").index
uav2_r4_z_force_idx = controller.get_sensor_info("uav_2_force_sensor_rotor_4_z").index

uav1_r4_z_force_idx = controller.get_sensor_info("uav_1_force_sensor_rotor_4_z").index


uav_2_joint_force_torque_idx = controller.get_sensor_info("uav_2_jft_sensor").index


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
uav_1_state_list = []
uav_2_state_list = []

rel_state_vector_list = []

dw_joint_sensor_readings = []
dw_body_sensor_readings = []
uav_2_jt_forces_list = []  # list of (x,y,z) force tuples 

uav_1_external_force_rotor4 = []

# each iteration sends control signal
while curr_sim_time < sim_max_duration:

    # log everything at current timestep  first
    time_seq.append(curr_sim_time)
    print("## Sim time:", curr_sim_time, "/", sim_max_duration, "s",
          " ## Sim steps:", curr_step ,"/", total_sim_steps, "steps")
    
    # Create controller input. This is the actuator input vector

    # uav 2 flies to right after 2.6 seconds
    if curr_sim_time < 25.0:
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
    uav1_pos_vel_xyz = read_sensor(reply, imu1_index, [0,1,2,6,7,8])
    uav2_pos_vel_xyz = read_sensor(reply, imu2_index, [0,1,2,6,7,8])

    uav_2_baselink_xyz_forces = read_multiple_sensors(reply, [fx_sensor_idx, fy_sensor_idx, fz_sensor_idx])
    uav_2_baselink_z_force = uav_2_baselink_xyz_forces[2]
    uav_2_rotors_z_forces = read_multiple_sensors(reply, [uav2_r1_z_force_idx, uav2_r2_z_force_idx,
                                                          uav2_r3_z_force_idx, uav2_r4_z_force_idx])
    uav_2_total_z_force = np.sum(np.append(uav_2_rotors_z_forces, uav_2_baselink_z_force))


    uav_2_jt_force = reply.get_sensor_output(uav_2_joint_force_torque_idx)
    uav_2_jt_forces_list.append(np.array(
                                        [
                                            uav_2_jt_force[0],
                                            uav_2_jt_force[1], 
                                            -(uav_2_jt_force[2] - 29.77)
                                        ]))
    
    uav_1_rotor_4_external_force = read_sensor(reply, uav1_r4_z_force_idx)
    uav_1_external_force_rotor4.append(uav_1_rotor_4_external_force)
    
   
    # evaluate both methods:
    print("Force on UAV 2 fixed joint:", -uav_2_jt_force[2] - 29.77)
    print("force on UAV 2 external_forces_sensor:", uav_2_total_z_force)

    dw_joint_sensor = uav_2_jt_force[2] - 29.77
    dw_joint_sensor_readings.append(-dw_joint_sensor)
    dw_body_sensor = uav_2_total_z_force
    dw_body_sensor_readings.append(dw_body_sensor)
    #print("#######")
    #print(dw_joint_sensor)
    #print(dw_body_sensor)


    # add x,y,z positions of uav 1
    uav_1_state_list.append(np.array(uav1_pos_vel_xyz))
    # add x,y,z positions of uav 2
    uav_2_state_list.append(np.array(uav2_pos_vel_xyz))
    
    # add z force of uav 2
    rel_state_vector = np.round(uav_1_state_list[-1] - uav_2_state_list[-1],2)
    print(rel_state_vector)
    rel_state_vector_list.append(rel_state_vector)
    

    # advance timer and step counter:
    curr_sim_time += steps_per_call * time_step
    curr_step += steps_per_call


# Clear and close simulator
print("Finished, clearing...")
controller.clear()
controller.close()
print("Control experiment ended.")

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