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

nan = float('NaN')
planner = Planner(velocity=1.5, acceleration_time=4.5, end=(16,0,0))
#planner = Planner(circular=True, velocity=1.2, step_size=0.1, acceleration_time=1.0, radius=2.0)
nfc = NonlinearFeedbackController()


def main(controller):
    

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


            
            positions.append(pos); velocities.append(vel); planned_pos.append(current_waypoint)
            print("position:",positions[-1])


            desired_waypoint, desired_yaw = current_waypoint[:9], current_waypoint[9]
            nfc.target_yaw = desired_yaw
            px4_input = nfc.nonlinear_feedback_qt_px4(desired_waypoint)

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


