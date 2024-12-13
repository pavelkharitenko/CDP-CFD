from simcontrol import simcontrol2
from controller import NonlinearFeedbackController
import numpy as np
np.set_printoptions(precision=2)
np.set_printoptions(suppress=True)

import numpy as np
import matplotlib.pyplot as plt
import sys, os
#sys.path.append('../../observers/baseline/')
#sys.path.append('../../uav/')
sys.path.append('../../utils/')
from uav import *
from utils import *

#rot = 1.570796326794897

nfc = NonlinearFeedbackController()
#nfc.target_yaw = rot
Fg = -9.81*3.035
#target_acc = [0.0, 0.0, 0.1]

target_waypoint = np.array([0.0,0.0,3.0, 
                            0.0,0.0,0.0,
                            0.0,0.0,0.0]) # pos, acc, vel


def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')



def compute_torques(omega_0, omega_1, omega_2, omega_3, k, d):
        # roll torque (τ_x)
        tau_x = k * d * (omega_0**2 + omega_1**2 - omega_2**2 - omega_3**2)
        # pitch torque (τ_y)
        tau_y = k * d * (omega_3**2 + omega_1**2 - omega_0**2 - omega_2**2)
        # yaw torque (τ_z)
        tau_z = -k * (omega_1**2 - omega_0**2 + omega_3**2 - omega_2**2)
    
        return tau_x, tau_y, tau_z


def main(controller):

    nan = float('NaN')
    mass = 3.035 # P600 weight

    target_rot = 0.0

    

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

    hovertime = 2.0
    throttle = 0.3347 # 0.3325 too low, 0.335 too high

    # Start simulation
    controller.clear()
    controller.start()

    # Retrive simulation time step (this should be after simulation start)
    time_step = controller.get_time_step()

    # Calculate time steps between one control period
    control_frequency = 200.0
    steps_per_call = int(1.0 / control_frequency / time_step)

    # Initialize timer
    t = 0.0
    measured_a_zs = []
    avg_rps_list = []

    # Simulation loop, simulating for 20 seconds (simulation time, not physical time)
    while (t < 10.0):
        # Set px4 control input to do position tracking of a circle trajectory
        # See actuator.md for details of input
        print(t)
        if t > hovertime:

            #target_rot += 0.02


            
            nfc.target_yaw = target_rot * np.pi/180

            roll, pitch, yaw, thrust = nfc.nonlinear_feedback(target_waypoint)
            roll, pitch, yaw = np.array([roll, pitch, yaw]) * 180.0/np.pi

            # todo: mismatch rotation attitude and highlevel position controller
            # print and visualize both
            qw, qx, qy, qz = euler_angles_to_quaternion(roll, pitch, yaw)
            throttle = thrust * nfc.TWR

            print("thrust from attitude controller", thrust)
            #print("throttle set:", throttle)
            px4_input = (1.0, qw, qx, qy, qz, 0.0, throttle) # This is the actuator input vector



        else:
            px4_input = (0.0, 0.0, 0.0, 0.5, nan, nan, nan, nan, nan, nan, 0.0, nan) 
            #px4_input = (0.0, nan, nan, nan, 0.0, 1.0, 0.0, nan, nan, nan, 1.570796326794897, nan) 


        # Simulate a control period, giving actuator input and retrieving sensor output
        reply = controller.simulate(steps_per_call,  {px4_index: px4_input})
        imu_data = reply.get_sensor_output(imu_index)

        pos = imu_data[:3]
        print("current pos", pos)
        vel = imu_data[3:6]
        acc = imu_data[6:9]

        nfc.feedback(pos, vel, acc)


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
        #if np.round(t,1) % .1 == 0.0:

        #print("Time", t)
        #print("Avg rotor speed", avg_rps_list[-1])
        #print("Predicted tot. Thrust with fitted function:", rps_to_thrust_p005_mrv80(avg_rps_list[-1]))
        Ct =  0.000362
        rho = 1.225
        A = 0.11948
        k = Ct * rho * A
        d = 0.3
        F = k * np.mean(np.abs(avg_rps_list[-1])**2)

        tau_x, tau_y, tau_z = compute_torques(r1_rps, r2_rps, r3_rps, r4_rps, k, d)

        
        
        #print("####### Rotor speed:", np.mean(np.abs(avg_rps_list[-1])))
        # Display the results
        #print(f"Computed Total Thrust from constants: {4*F} N")
        #print(f"Roll Torque (τx) {tau_x} Nm")
        #print(f"Pitch Torque (τy): {tau_y} Nm")
        #print(f"Yaw Torque (τz): {tau_z} Nm")
        #print("------------")
           # print("Roll torque imu:", )


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


