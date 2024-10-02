from simcontrol import simcontrol2
import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.append('../../observers/baseline/')
from utils import *

port = 25556

SIM_DURATION = 30.0

DRONE_TOTAL_MASS = 3.035 # P600 weight

HOVER_TIME = 30.0
FLY_CIRCULAR = False
freq = 0.02
radius = 3.0


def read_sensor(reply, sensor_idx, indicies=[0]):
    result = []
    sensor_tuple_data = reply.get_sensor_output(sensor_idx)

    for i in indicies:
        result.append(sensor_tuple_data[i])
    if len(result) == 1:
        return result[0]
    else:
        return result

def read_multiple_sensors(reply, sensor_indices, tuple_indices=None):
    result = []
    if not tuple_indices:
        for sensor_idx in sensor_indices:
            result.append(read_sensor(reply, sensor_idx))
    else:
        for sensor_idx in sensor_indices:
            result.append(read_sensor(reply, sensor_idx, tuple_indices))
    return result

def get_sensor_idx(sensor_names):
    if isinstance(sensor_names, list):
        result = []
        for sensor_name in sensor_names:
            result.append(controller.get_sensor_info(sensor_name).index)
        return result
    else:
        return controller.get_sensor_info(sensor_names).index


# connect to simulators controller
controller = simcontrol2.Controller("localhost", port)
# retrieve px4 actuator index:
px4_index_1 = controller.get_actuator_info('controller1').index


# setup sensors
imu1_idx, imu2_idx = get_sensor_idx(["imu1", "imu2"])
# UAV 2
# uav 2 xyz body external force sensors
fx_sensor_idx, fy_sensor_idx, fz_sensor_idx = get_sensor_idx(["uav_2_force_sensor_z", "uav_2_force_sensor_z","uav_2_force_sensor_z"])
# uav 2 external z force sensors:
uav2_r1234_z_force_idx = get_sensor_idx(["uav_2_force_sensor_rotor_1_z", "uav_2_force_sensor_rotor_2_z",
                                         "uav_2_force_sensor_rotor_3_z", "uav_2_force_sensor_rotor_4_z"])
# uav 2 joint force torque
uav_2_joint_force_torque_idx = controller.get_sensor_info("uav_2_jft_sensor").index

# UAV 1
# uav 1 external z force sensors:
uav_1_body_r1234_ext_z_force_idx = get_sensor_idx(["uav_1_force_sensor_body_z", "uav_1_force_sensor_rotor_1_z",
                                               "uav_1_force_sensor_rotor_2_z", "uav_1_force_sensor_rotor_3_z",
                                               "uav_1_force_sensor_rotor_4_z"])
# uav 1 jointforcetorque sensors 
uav_1_jft_sensor_idxs = get_sensor_idx(["uav_1_jft_sensor_body", "uav_1_jft_sensor_imu",
                                        "uav_1_jft_sensor_rotor1", "uav_1_jft_sensor_rotor2",
                                        "uav_1_jft_sensor_rotor3", "uav_1_jft_sensor_rotor4"])



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

# uav 2 measurements
uav_2_state_list = []

dw_joint_sensor_readings = []
dw_body_sensor_readings = []
uav_2_jt_forces_list = []  # list of (x,y,z) force tuples 

# uav 1 measurements
uav_1_state_list = []


uav_1_ext_force_body_r1234_list = []


uav_1_jtf_list = [] # contains lists of tuples: [(fx,fy,fz,tor_x,tor_y,tor_z), (fx,fy,fz,tor_x,tor_y,tor_z), ..., x5]
uav1_acc_xyz_list = []

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
            


    # Simulate a control period, giving actuator input and retrieving sensor output
    reply = controller.simulate(steps_per_call,  {
                                                   px4_index_1: px4_input_1, 
                                                  #px4_index_2: px4_input_2
                                                  })
    
    

    # Check simulation result
    if reply.has_error():
        print('Simulation failed! Terminating control experiement.')
        break

    
    # Get imu sensor output (a tuple of floats) of xyz pos, vel, acc
    uav1_pos_vel_acc_xyz = read_sensor(reply, imu1_idx, [0,1,2,6,7,8,12,13,14])
    uav2_pos_vel_acc_xyz = read_sensor(reply, imu2_idx, [0,1,2,6,7,8,12,12,14])




    # Get uav 2 dw forces
    uav_2_baselink_xyz_forces = read_multiple_sensors(reply, [fx_sensor_idx, fy_sensor_idx, fz_sensor_idx])
    uav_2_baselink_z_force = uav_2_baselink_xyz_forces[2]
    uav_2_rotors_z_forces = read_multiple_sensors(reply, uav2_r1234_z_force_idx)
    uav_2_total_z_force = np.sum(np.append(uav_2_rotors_z_forces, uav_2_baselink_z_force))


    uav_2_jt_force = reply.get_sensor_output(uav_2_joint_force_torque_idx)
    uav_2_jt_forces_list.append(np.array([
                                            uav_2_jt_force[0],
                                            uav_2_jt_force[1], 
                                            -(uav_2_jt_force[2] - 29.77)
                                        ]))
    
    
   
    # evaluate both methods:
    #print("Force on UAV 2 fixed joint:", -uav_2_jt_force[2] - 29.77)
    #print("force on UAV 2 external_forces_sensor:", uav_2_total_z_force)

    dw_joint_sensor = uav_2_jt_force[2] - 29.77
    dw_joint_sensor_readings.append(-dw_joint_sensor)
    dw_body_sensor = uav_2_total_z_force
    dw_body_sensor_readings.append(dw_body_sensor)

    # read uav 1 sensors
    # uav 1 rotor external force 
    uav_1_ext_force_body_r1234 = read_multiple_sensors(reply, uav_1_body_r1234_ext_z_force_idx)
    uav_1_ext_force_body_r1234_list.append(uav_1_ext_force_body_r1234)


    # uav 1 body and rotors jft sensors
    uav1_body_r1_r2_r3_r4_jft = read_multiple_sensors(reply, uav_1_jft_sensor_idxs, [0,1,2])

    uav_1_jtf_list.append(uav1_body_r1_r2_r3_r4_jft)

    # add x,y,z positions of uav 1
    uav_1_state_list.append(np.array(uav1_pos_vel_acc_xyz))
    # add x,y,z positions of uav 2
    uav_2_state_list.append(np.array(uav2_pos_vel_acc_xyz))
    
    # add z force of uav 2
    rel_state_vector = np.round(uav_1_state_list[-1] - uav_2_state_list[-1], 2)
    #print(rel_state_vector)
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

plot_3d_vectorfield(rel_state_vector_list, uav_2_jt_forces_list, 1.0/np.max(np.abs(uav_2_jt_forces_list)), "\n Collected forces")

plot_uav_force_statistics(time_seq, uav_1_ext_force_body_r1234_list, uav_1_jtf_list, 
                          [DRONE_TOTAL_MASS*state[8] for state in uav_1_state_list])

plt.show()

exit(0)
plot_static_dw_collection(uav_2_state_list, uav_1_state_list, dw_body_sensor_readings, uav_2_jt_forces_list, time_seq)



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

exit(0)

ax3 = axes[0][2]
ax3.set_title('Producer UAV External Z force on rotor 4')
ax3.set_xlabel('time (s)')
ax3.set_ylabel('Force (N)')
#ax3.plot(time_seq, , label='body extfor z')
ax3.plot(time_seq, uav_1_external_force_rotor1, label='r1 extfor z')
ax3.plot(time_seq, uav_1_external_force_rotor2, label='r2 extfor z')
ax3.plot(time_seq, uav_1_external_force_rotor3, label='r3 extfor z')
ax3.plot(time_seq, uav_1_external_force_rotor4, label='r4 extfor z')
ax3.plot(time_seq, [np.sum(xyz) for xyz in zip(uav_1_external_force_body, uav_1_external_force_rotor1, uav_1_external_force_rotor2,
                                              uav_1_external_force_rotor3, uav_1_external_force_rotor4)], label='sum extfor z')


#ax3.plot(time_seq, [xyz[1] for xyz in uav_2_jt_forces_list], label='force_y')
ax3.legend()




ax3 = axes[1][2]
ax3.set_title('Producer UAV jft Z forces')
ax3.set_xlabel('time (s)')
ax3.set_ylabel('Force (N)')

ax3.plot(time_seq, [body_r1_r2_r3_r4[0][2] for body_r1_r2_r3_r4 in uav_1_jtf_list], label='jft body z')
ax3.plot(time_seq, [body_r1_r2_r3_r4[1][2] for body_r1_r2_r3_r4 in uav_1_jtf_list], label='jft imu z')
ax3.plot(time_seq, [-body_r1_r2_r3_r4[2][2] for body_r1_r2_r3_r4 in uav_1_jtf_list], label='jft r1 z')
ax3.plot(time_seq, [-body_r1_r2_r3_r4[3][2] for body_r1_r2_r3_r4 in uav_1_jtf_list], label='jft r2 z')
ax3.plot(time_seq, [-body_r1_r2_r3_r4[4][2] for body_r1_r2_r3_r4 in uav_1_jtf_list], label='jft r3 z')
ax3.plot(time_seq, [-body_r1_r2_r3_r4[5][2] for body_r1_r2_r3_r4 in uav_1_jtf_list], label='jft r4 z')
fz_fu_diff = np.array([DRONE_TOTAL_MASS *acc_xyz[2]  for acc_xyz in uav1_acc_xyz_list]) - np.array([-body_r1_r2_r3_r4[2] - DRONE_TOTAL_MASS*9.81 for body_r1_r2_r3_r4 in np.sum(uav_1_jtf_list, axis=1)]) 
ax3.plot(time_seq, [-body_r1_r2_r3_r4[2] for body_r1_r2_r3_r4 in np.sum(uav_1_jtf_list, axis=1)], label='jft z in total')
ax3.plot(time_seq, [DRONE_TOTAL_MASS *acc_xyz[2]  for acc_xyz in uav1_acc_xyz_list], label='actual z force')
ax3.plot(time_seq, fz_fu_diff, label='residual z force')



#ax3.plot(time_seq, [xyz[1] for xyz in uav_2_jt_forces_list], label='force_y')
ax3.legend()



ax3 = axes[2][2]
ax3.set_title('Producer UAV Body Jft forces')
ax3.set_xlabel('time (s)')
ax3.set_ylabel('Force (N)')

ax3.plot(time_seq, [body[0] for body in uav_1_body_jft_list], label='jft body x')
ax3.plot(time_seq, [body[1] for body in uav_1_body_jft_list], label='jft body y')
ax3.plot(time_seq, [body[2] for body in uav_1_body_jft_list], label='jft body z')


#ax3.plot(time_seq, [xyz[1] for xyz in uav_2_jt_forces_list], label='force_y')
ax3.legend()

axes[0][0].set_visible(False)
axes[1][0].set_visible(False)
axes[2][0].set_visible(False)
axes[2][2].set_visible(False)


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




