from simcontrol import simcontrol2
from datetime import datetime
import keyboard
import numpy as np
import matplotlib.pyplot as plt
import sys, time, randomname, pickle
from numpy import random as rnd

sys.path.append('../../uav/')
sys.path.append('../../utils/')
#sys.path.append('../../../../../notify/')
#from notify_script_end import notify_ending

from uav import *
from utils import *

def main(controller):

    SAVE_EXP = True
    SAVE_INTERVALL = 60.0

    port = 25557
    SIM_MAX_DURATION = 720.0
    HOVER_TIME = 5.0
    DRONE_TOTAL_MASS = 3.035 # P600 weight
    
    nan = float('NaN')


    exp_name = init_experiment("Precise-collection")

    EXP_SUCCESSFUL = True  # is set to false when errors occur


    # connect to simulators controller
    controller = simcontrol2.Controller("localhost", port)
    ext_force_sensors = ["force_sensor_body_z", "force_sensor_rotor_1_z", "force_sensor_rotor_2_z", "force_sensor_rotor_3_z", "force_sensor_rotor_4_z"]
    jft_sensors = ["jft_sensor_body", "jft_sensor_imu",  "jft_sensor_rotor1", "jft_sensor_rotor2", "jft_sensor_rotor3", "jft_sensor_rotor4"]
    joint_sensors = ["joint_sensor_r1", "joint_sensor_r2", "joint_sensor_r3", "joint_sensor_r4",]



    # setup UAVs
    uav_1_ext_z_force_sensors = ["uav_1_" + ext_sen_name for ext_sen_name in ext_force_sensors]
    uav_2_ext_z_force_sensors = ["uav_2_" + ext_sen_name for ext_sen_name in ext_force_sensors]
    uav_1_jft_sensors = ["uav_1_" + ext_sen_name for ext_sen_name in jft_sensors]
    uav_2_jft_sensors = ["uav_2_" + ext_sen_name for ext_sen_name in jft_sensors]
    uav_1_joint_sensors = ["uav_1_" + joint_sen for joint_sen in joint_sensors]
    uav_2_joint_sensors = ["uav_2_" + joint_sen for joint_sen in joint_sensors]


    uav_1 = UAV("producer", controller, "controller1", "imu1", uav_1_ext_z_force_sensors, uav_1_jft_sensors, uav_1_joint_sensors)
    uav_2 = UAV("sufferer", controller, "controller2", "imu2", uav_2_ext_z_force_sensors, uav_2_jft_sensors, uav_2_joint_sensors)



    # Start simulation
    controller.clear()
    controller.start()
    time_step = controller.get_time_step()  # time_step = 0.0001
    
    total_sim_steps = SIM_MAX_DURATION / time_step
    control_frequency = 400.0 # Hz
    # Calculate time steps between one control period
    steps_per_call = int(1.0 / control_frequency / time_step)
    print("One Timestep is ", time_step, "||", "Steps per call are", steps_per_call,"||", "Sim dt is timestep * steps per call =", time_step * steps_per_call)


    # Initialize timer
    curr_sim_time = 0.0
    curr_step = 0
    time_seq = []
    rel_state_vector_list = []
   
    next_way_point_uav_1 = [0.0,-1.0,0.7] # uav 1 hover at (0,-1,0.7)
    yaw_uav_1 = 1.570796326794897


    # each iteration sends control signal
    while curr_sim_time < SIM_MAX_DURATION:

        
        time_seq.append(curr_sim_time) # log everything at current timestep  first
        #print("### Sim time:", np.round(curr_sim_time,3), "/", SIM__MAX_DURATION, "s", " ## Sim steps:", curr_step ,"/", total_sim_steps, "steps ###")


        current_time = np.round(curr_sim_time, 3)
        if current_time >= HOVER_TIME and current_time % 4.000 == 0.0:
            next_way_point_uav_1 = sample_3d_point(uav_1.states[-1][1])
            #print("next point set to", next_way_point_uav_1)
        
        px4_input_1 = (0.0, next_way_point_uav_1[0], next_way_point_uav_1[1], next_way_point_uav_1[2], nan, nan, nan, nan, nan, nan, yaw_uav_1, nan) 
        px4_input_2 = (0.0, 0, 0.0, 0.0, nan, nan, nan, nan, nan, nan, 0.0, nan) # uav 2 hover at (0,0,0)
            

        # Simulate a control period, giving actuator input and retrieving sensor output
        reply = controller.simulate(steps_per_call,  { uav_1.px4_idx: px4_input_1, uav_2.px4_idx: px4_input_2 })
        
        # Check simulation result
        if reply.has_error():
            print('Simulation failed! Terminating control experiement.')
            #notify_ending("script failed at time " + str(curr_sim_time))
            EXP_SUCCESSFUL = False
            break
        
        # read sensors of uavs
        uav_1.update(reply, curr_sim_time)
        uav_2.update(reply, curr_sim_time)

        rel_state_vector_uav_2 = np.round(np.array(uav_1.states[-1]) - np.array(uav_2.states[-1]), 3)
        #print("Rel. state vec.:", rel_state_vector_uav_2)
        rel_state_vector_list.append(rel_state_vector_uav_2)


        # log state of experiment:
        if current_time >= HOVER_TIME and current_time % 15.000 == 0.0:
            print("### -----------------------------------------")
            print("### Sim time:", curr_sim_time, "/", SIM_MAX_DURATION, "s", " ## Sim steps:", curr_step ,"/", total_sim_steps, "steps ###")
            print("Collected ", len(time_seq), "samples of data.")
            print("Max. force on uav 2 recorded: ", np.max(np.abs(np.array(uav_2.states)[:,8]*DRONE_TOTAL_MASS)))
            print("Max. Y-vel. of uav 1 recorded: ", np.max(np.abs(np.array(uav_1.states)[:,4])))
            print("Min height of uav 1:", np.min(np.array(uav_1.states)[:,2]))
            print("Min height of uav 2:", np.min(np.array(uav_2.states)[:,2]))

        if current_time >= HOVER_TIME and SAVE_EXP and current_time % SAVE_INTERVALL == 0.0:
            uav_1.controller, uav_2.controller = None, None
            exp_path = save_experiment(exp_name + "intermediate", [uav_1, uav_2], True,  current_time)
            uav_1.controller, uav_2.controller = controller, controller
            print("### intermediate saved to ", exp_path)


        
        
        # advance timer and step counter:
        curr_sim_time += steps_per_call * time_step
        curr_step += steps_per_call


    # Clear and close simulator
    print("Finished, clearing...")
    controller.clear()
    controller.close()
    print("Control loop ended.")
    print("Collected ", len(time_seq), "samples of data.")

    if SAVE_EXP:
        uav_1.controller, uav_2.controller = None, None
        exp_path = save_experiment(exp_name, [uav_1, uav_2], True,  SIM_MAX_DURATION)
        print("### Finally saving experiment to ", exp_path)



    





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



