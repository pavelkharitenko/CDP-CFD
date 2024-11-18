# testing noise filtering techniques, single uav taking off and hovering
from simcontrol import simcontrol2
import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.append('../../observers/baseline/')
sys.path.append('../../uav/')
sys.path.append('../../utils/')
from uav import *
from utils import *


def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

def main(controller):

    nan = float('NaN')
    mass = 3.035 # P600 weight

    # Create simulation controller
    controller = simcontrol2.Controller('localhost', 25556)

    # Retrieve px4 actuator index (px4 is a class of actuator)
    px4_index = controller.get_actuator_info('controller').index

    # Retrieve imu sensor index (imu is a class of sensor)
    imu_index = controller.get_sensor_info('imu').index
    r1_joint_sensor_idx = controller.get_sensor_info("uav_2_r1_joint_sensor").index
    r2_joint_sensor_idx = controller.get_sensor_info("uav_2_r2_joint_sensor").index
    r3_joint_sensor_idx = controller.get_sensor_info("uav_2_r3_joint_sensor").index
    r4_joint_sensor_idx = controller.get_sensor_info("uav_2_r4_joint_sensor").index

    hovertime = 0.0
    throttle = 0.362

    # Start simulation
    controller.clear()
    controller.start()

    # Retrive simulation time step (this should be after simulation start)
    time_step = controller.get_time_step()

    # Calculate time steps between one control period
    control_frequency = 400.0
    steps_per_call = int(1.0 / control_frequency / time_step)

    # Initialize timer
    t = 0.0
    measured_a_zs = []
    avg_rps_list = []

    # Simulation loop, simulating for 20 seconds (simulation time, not physical time)
    while (t < 20.0):
        # Set px4 control input to do position tracking of a circle trajectory
        # See actuator.md for details of input
        
        if t > hovertime:

            #px4_input = (2.0, 0.0, 0.0, 0.0, throttle) # This is the actuator input vector
            px4_input = (0.0, 0.0, 0.0, 1.0, nan, nan, nan, nan, nan, nan, 0.0, nan) 

        else:
            px4_input = (0.0, 0.0, 0.0, 1.0, nan, nan, nan, nan, nan, nan, 0.0, nan) 

        # Simulate a control period, giving actuator input and retrieving sensor output
        reply = controller.simulate(steps_per_call,  {px4_index: px4_input})
        imu_data = reply.get_sensor_output(imu_index)
        a_z = imu_data[14]
        measured_a_zs.append(a_z)

        r1_rps = reply.get_sensor_output(r1_joint_sensor_idx)[0]
        r2_rps = reply.get_sensor_output(r2_joint_sensor_idx)[0]
        r3_rps = reply.get_sensor_output(r3_joint_sensor_idx)[0]
        r4_rps = reply.get_sensor_output(r4_joint_sensor_idx)[0]

        avg_rps_list.append((np.abs(r1_rps) + np.abs(r2_rps) + np.abs(r3_rps) + np.abs(r4_rps))/4)

        # Advance timer
        t += steps_per_call * time_step


        
        #print(f'x: {p_x:.2f}, y: {p_y:.2f}, z: {p_z:.2f}')
        
        print(t)

    # Clear simulator
    print("Finished, clearing...")
    controller.clear()

    # Close the simulation controller
    controller.close()
    print("Done.")

    print("max. recorded rps:", np.max(avg_rps_list))
    print("min. recorded rps:", np.min(avg_rps_list))


    thrusts = [rps_to_thrust_p005_mrv80(avg_rps) - 9.81*mass for avg_rps in avg_rps_list]
    uav_z_forces = np.array(measured_a_zs)*mass

    fig = plt.subplot()
    fig.plot(moving_average(uav_z_forces, 20), label="UAV's z-axis forces")
    fig.plot(thrusts, label="controller's \n rps forces")
    plt.legend()
    plt.show()

    ignore_first_k = 200
    data = uav_z_forces[ignore_first_k:]
    # Parameters for the running average and variance
    window_size = 20  # Adjust the window size as needed

    # Compute running average and running variance
    running_avg = np.convolve(data, np.ones(window_size) / window_size, mode='valid')
    running_var = [np.var(data[i:i + window_size]) for i in range(len(data) - window_size + 1)]
    running_std_dev = np.sqrt(running_var)
    # compute total variance
    total_var = np.full(len(running_avg),np.var(running_avg))
    total_std = np.sqrt(total_var)
    # Define the x-axis for the running metrics
    x_vals = np.arange(window_size - 1, len(data))

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(data, label="IMU Measurements", linewidth=0.4)

    # Plot the variance tube (1 standard deviation above and below the running average)
    plt.fill_between(x_vals, 
                     running_avg - total_std, 
                     running_avg + total_std, 
                    color='orange', 
                    alpha=0.4, 
                    label="Variance (±1 STD)")
    
    plt.plot(x_vals, running_avg, color='orange', label="Running Average", linewidth=2)

    # adjust length of input u to averaged measurements:
    thrusts = thrusts[ignore_first_k:]
    plt.plot(x_vals, thrusts[window_size-1:len(data)], label="controller's input thrust", linewidth=2, color="magenta")


    # Labels and legend
    plt.xlabel("Timesteps")
    plt.ylabel("Value")
    plt.title("Raw IMU measurement and its running average for smoother measurements")
    plt.legend()
    plt.show()

    
    
controller = None
try:
    main(controller)
except KeyboardInterrupt as Exp:
    print("----------Error encountered:----------")
    print(Exp)
    print("--------------------")

    if controller:
        print("closing controller")
        controller.clear()
        controller.close()


