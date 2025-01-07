from simcontrol import simcontrol2
from controller import NonlinearFeedbackController
import numpy as np
np.set_printoptions(precision=2)
np.set_printoptions(suppress=True)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
import sys, os
#sys.path.append('../../observers/baseline/')
#sys.path.append('../../uav/')
sys.path.append('../../utils/')
from planner import Planner

from utils import *


planner = Planner(velocity=1.5, acceleration_time=2.7, end=(16,0,0), circular=True)
#planner = Planner(circular=True, velocity=1.2, step_size=0.1, acceleration_time=1.0, radius=2.0)
nfc = NonlinearFeedbackController()





def main(controller):

    nan = float('NaN')
    mass = 3.035 # P600 weight

    target_rot = 20.0

    

    # Create simulation controller
    controller = simcontrol2.Controller('localhost', 25556)

    # Retrieve px4 actuator and imu index
    px4_index = controller.get_actuator_info('controller').index
    imu_index = controller.get_sensor_info('imu').index


    hovertime = 0.0

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
    velocities = []
    planned_pos = []

    current_waypoint = planner.pop_waypoint()
    print("first waypoint at", current_waypoint)
    #exit(0)

    # Simulation loop, simulating for 20 seconds (simulation time, not physical time)
    while (t < 7.0):
        # Set px4 control input to do position tracking of a circle trajectory
        # See actuator.md for details of input
        if t > hovertime:
            print(" ------------------ ")
            print("t:",t)


            
            positions.append(pos)
            velocities.append(vel)
            planned_pos.append(current_waypoint)
            print("position:",positions[-1])


            desired_waypoint, yaw = current_waypoint[:9], current_waypoint[9]
            nfc.target_yaw = yaw
            


            roll, pitch, yaw, thrust = nfc.nonlinear_feedback_nf(desired_waypoint)
            roll, pitch, yaw = np.array([roll, pitch, yaw]) * 180.0/np.pi # convert to radians

            qw, qx, qy, qz = euler_angles_to_quaternion(roll, pitch, yaw)
            throttle = thrust * nfc.TWR
            px4_input = (1.0, qw, qx, qy, qz, 0.0, throttle) # This is the actuator input vector

        else:
            px4_input = (0.0, 0.0, 0.0, 0.0, nan, nan, nan, nan, nan, nan, 0.0, nan) 


        # Simulate a control period, giving actuator input and retrieving sensor output
        reply = controller.simulate(steps_per_call,  {px4_index: px4_input})
        imu_data = reply.get_sensor_output(imu_index)

        pos = imu_data[:3]
        vel = imu_data[6:9]
        acc = imu_data[12:15]

        nfc.feedback(pos, vel, acc)

        # check if current target already reached
        if np.linalg.norm(np.array(pos[:2]) - current_waypoint[:2])<0.15:
            current_waypoint = planner.pop_waypoint()

        # Advance timer
        t += steps_per_call * time_step



    # Clear simulator
    print("Finished, clearing...")
    controller.clear()

    # Close the simulation controller
    controller.close()
    print("Done.")

    

    import seaborn as sns
    from sklearn.metrics import mean_squared_error
    def plot_trajectory_analysis(actual_positions, planned_positions, actual_velocities):
        sns.set(style="whitegrid")

        # Calculate errors and RMSE
        position_errors = planned_positions[:, :3] - actual_positions
        velocity_errors = planned_positions[:, 3:6] - actual_velocities
        
        avg_position_errors = np.mean(np.abs(position_errors), axis=0)
        avg_velocity_errors = np.mean(np.abs(velocity_errors), axis=0)
        
        radial_position_errors = np.sqrt(position_errors[:, 0]**2 + position_errors[:, 1]**2)
        avg_radial_position_error = np.mean(radial_position_errors)

        fig, axes = plt.subplots(3, 2, figsize=(18, 15))
        
        # Z-Position Plot
        sns.lineplot(x=np.arange(len(actual_positions)), y=actual_positions[:, 2], ax=axes[0, 0], label='Actual Z', color='blue')
        sns.lineplot(x=np.arange(len(planned_positions)), y=planned_positions[:, 2], ax=axes[0, 0], label='Target Z', color='green')
        axes[0, 0].set_title('Z-Position')
        axes[0, 0].set_ylabel('Z Position (m)')
        
        # Z-Error Plot
        sns.lineplot(x=np.arange(len(position_errors)), y=position_errors[:, 2], ax=axes[1, 0], label='Z-Error', color='red')
        axes[1, 0].axhline(avg_position_errors[2], linestyle='--', color='orange', label=f'Avg Error ({avg_position_errors[2]:.2f})')
        axes[1, 0].set_title('Z-Position Error')
        axes[1, 0].set_ylabel('Error (m)')
        axes[1, 0].legend()
        
        # XYZ Velocities Plot
        for i, label in enumerate(['X', 'Y', 'Z']):
            sns.lineplot(x=np.arange(len(actual_velocities)), y=actual_velocities[:, i], ax=axes[0, 1], label=f'{label}-Velocity', color=['blue', 'green', 'red'][i])
        axes[0, 1].set_title('XYZ Velocities')
        axes[0, 1].set_ylabel('Velocity (m/s)')
        axes[0, 1].legend()
        
        # XYZ Velocity Error Plot
        for i, label in enumerate(['X', 'Y', 'Z']):
            sns.lineplot(x=np.arange(len(velocity_errors)), y=velocity_errors[:, i], ax=axes[1, 1], label=f'{label}-Velocity Error', color=['blue', 'green', 'red'][i])
        axes[1, 1].axhline(avg_velocity_errors[i], linestyle='--', color='orange', label=f'Avg Error ({avg_velocity_errors[i]:.2f})')
        axes[1, 1].set_title('XYZ Velocity Errors')
        axes[1, 1].set_ylabel('Error (m/s)')
        axes[1, 1].legend()
        
        # Statistics Bar Plot for Position Errors
        axes[2, 0].bar(['X Error', 'Y Error', 'Z Error', 'Radial Error'], 
                        [avg_position_errors[0], avg_position_errors[1], avg_position_errors[2], avg_radial_position_error], 
                        color=['blue', 'green', 'red', 'orange'])
        axes[2, 0].set_title('Average Position Errors')
        axes[2, 0].set_ylabel('Error (m)')
        
        # Statistics Bar Plot for Velocity Errors
        axes[2, 1].bar(['X Velocity Error', 'Y Velocity Error', 'Z Velocity Error'], 
                        avg_velocity_errors, color=['blue', 'green', 'red'])
        axes[2, 1].set_title('Average Velocity Errors')
        axes[2, 1].set_ylabel('Error (m/s)')

        plt.tight_layout()
        plt.show()


    
    

    plot_trajectory_analysis(np.array(positions), np.array(planned_pos), np.array(velocities))
    
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


