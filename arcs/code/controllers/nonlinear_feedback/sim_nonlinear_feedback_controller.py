from simcontrol import simcontrol2
from controller import NonlinearFeedbackController
import numpy as np
np.set_printoptions(precision=2)
np.set_printoptions(suppress=True)

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
import sys, os
#sys.path.append('../../observers/baseline/')
#sys.path.append('../../uav/')
sys.path.append('../../utils/')

from utils import *



nfc = NonlinearFeedbackController()


target_waypoint = np.array([0.0,0.0,0.0, 
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

    target_rot = 20.0

    

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
    throttle = 0.3347 # 0.3325 too low, 0.335 too high

    # Start simulation
    controller.clear()
    controller.start()

    # Retrive simulation time step (this should be after simulation start)
    time_step = controller.get_time_step()

    # Calculate time steps between one control period
    control_frequency = 100.0
    steps_per_call = int(1.0 / control_frequency / time_step)

    # Initialize timer
    t = 0.0
    measured_a_zs = []
    avg_rps_list = []
    positions = []

    thrust_nfc = []
    thrust_ac = []

    

    # Simulation loop, simulating for 20 seconds (simulation time, not physical time)
    while (t < 6.0):
        # Set px4 control input to do position tracking of a circle trajectory
        # See actuator.md for details of input
        if t > hovertime:
            print(" ------------------ ")
            #print("t:",t)

            #target_rot += 0.02

            
            positions.append(pos)
            print("position:",positions[-1])

            nfc.target_yaw = target_rot * np.pi/180.0
            #nfc.target_yaw = 0.0


            roll, pitch, yaw, thrust = nfc.nonlinear_feedback_nf(target_waypoint)
            roll, pitch, yaw = np.array([roll, pitch, yaw]) * 180.0/np.pi
            
            if t > hovertime + 222.0:
                #roll, pitch, yaw, thrust = nfc.set_xyz_force(0.0,-29.77,3.0)
                pass
            
            #roll, pitch, yaw = np.array([0, 0, yaw]) * 180.0/np.pi
            



           
            qw, qx, qy, qz = euler_angles_to_quaternion(roll, pitch, yaw)
            throttle = thrust * nfc.TWR
            #print("Throttle for attitu. contrl.", throttle)

            print("thrust for attit. contrl.", thrust)
            #print("throttle set:", throttle)
            px4_input = (1.0, qw, qx, qy, qz, 0.0, throttle) # This is the actuator input vector



        else:
            px4_input = (0.0, 0.0, 0.0, 1.0, nan, nan, nan, nan, nan, nan, 0.0, nan) 
            #px4_input = (0.0, nan, nan, nan, 0.0, 1.0, 0.0, nan, nan, nan, 1.570796326794897, nan) 


        # Simulate a control period, giving actuator input and retrieving sensor output
        reply = controller.simulate(steps_per_call,  {px4_index: px4_input})
        imu_data = reply.get_sensor_output(imu_index)

        pos = imu_data[:3]
        vel = imu_data[6:9]
        acc = imu_data[12:15]

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

    
    
    plot_positions_with_references(positions, 
                                   x_refs=target_waypoint[0], 
                                   y_refs=target_waypoint[1], 
                                   z_refs=target_waypoint[2])

    
    analyze_and_plot_forces(avg_rps_list, measured_a_zs, mass, rps_to_thrust_p005_mrv80, 
                            ignore_first_k=200, window_size=30)
 
    
    
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


