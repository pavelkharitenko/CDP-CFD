from simcontrol import simcontrol2
import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append('../../../uav/')
sys.path.append('../../../utils/')
#sys.path.append('../../../../../notify/')
#from notify_script_end import notify_ending


from uav import *
from utils import *

def main(controller):
    # manuever specific
    MANUEVER_NAME = "1_flybelow"
    DRONE_TOTAL_MASS = 3.035 # P600 mass
    port = 25556
    nan = float('NaN')

    # exp specific
    SAVE_EXP = True
    dataset_name="demo_1_flybelow"
    SIM_MAX_DURATION = 80.0

    SAVE_INTERVALL = 800.0
    # episode specific
    HOVER_DURATION = 20.0
    ITERATION_TIME = HOVER_DURATION + 2.5
    exp_name = init_experiment(MANUEVER_NAME)
    total_iterations = 0
    selected_velocities = []

    # controller & uav init
    controller = simcontrol2.Controller("localhost", port)
    uav_1, uav_2 = init_two_uavs(controller)

    # Start simulation
    controller.clear()
    controller.start()
    time_step = controller.get_time_step()  # time_step = 0.0001
    
    total_sim_steps = SIM_MAX_DURATION / time_step
    control_frequency = 200.0 # Hz
    steps_per_call = int(1.0 / control_frequency / time_step)
    print("One Timestep is ", time_step, "||", "Steps per call are", steps_per_call,"||", "Sim dt is timestep * steps per call =", time_step * steps_per_call)


    # Initialize timer
    curr_sim_time = 0.0
    iteration_time = 0.0
    curr_step = 0
    time_seq = []
    rel_state_vector_list = []

   

    next_way_point_uav_1 = [0.0,-1.5,1.5] # uav 1 hover at (0,-1,0.7)
    yaw_uav_1 = 4.71238898038469
    yaw_uav_2 = 1.570796326794897
    # sample initial z-position for uav 2
    y_vel_max = 1.5#2.0
    y_vel_min = 1.5#0.5
    y_pos_max = 2.5
    initial_point = np.array((0.0,0.0,0.0))#sample_3d_point(1.0) # returns at negative side a point p=[N(), N(), N()]
    sampled_y_vel = sample_from_range(y_vel_min,y_vel_max)
    initial_y_pos = -2.5#-1.5 - sampled_y_vel*0.2
    selected_velocities.append(sampled_y_vel)
    ITERATION_TIME = HOVER_DURATION + 10.0/sampled_y_vel

    px4_input_1 = (0.0, 0.0, 0.0, 0.0, nan, nan, nan, nan, nan, nan, yaw_uav_1, nan) # uav 1 hover at (0,0,0)
    px4_input_2 = (0.0, initial_point[0], initial_y_pos, initial_point[2]-2.0, nan, nan, nan, nan, nan, nan, yaw_uav_2, nan) # uav 2 hover at (x,0,z)


    # each iteration sends control signal
    while curr_sim_time < SIM_MAX_DURATION:

        

        while iteration_time < HOVER_DURATION:
            # bring uavs first into initial position
            reply = controller.simulate(steps_per_call,  { uav_1.px4_idx: px4_input_1, uav_2.px4_idx: px4_input_2 })
            curr_sim_time += steps_per_call * time_step
            curr_step += steps_per_call
            iteration_time += steps_per_call * time_step

        # begin iteration
        #print("beginning iteration at global time",np.round(curr_sim_time,2))
        px4_input_2 = (0.0, nan, nan, nan, 0.0, sampled_y_vel, 0.0, nan, nan, nan, yaw_uav_2, nan)
        
            

        # Simulate a control period, giving actuator input and retrieving sensor output
        reply = controller.simulate(steps_per_call,  { uav_1.px4_idx: px4_input_1, uav_2.px4_idx: px4_input_2 })
        
        # Check simulation result
        if reply.has_error():
            print('Simulation failed! Terminating control experiement.')
            #notify_ending("script failed at time " + str(curr_sim_time))
            
            break
        
        time_seq.append(iteration_time - HOVER_DURATION) 
        uav_1.update(reply, iteration_time)
        uav_2.update(reply, iteration_time)

        rel_state_vector_uav_2 = np.round(np.array(uav_1.states[-1]) - np.array(uav_2.states[-1]), 3)
        #print("Rel. state vec.:", rel_state_vector_uav_2)
        rel_state_vector_list.append(rel_state_vector_uav_2)


        if SAVE_EXP and curr_sim_time % SAVE_INTERVALL == 0.0:
            uav_1.controller, uav_2.controller = None, None
            exp_path = save_experiment(exp_name + "intermediate", [uav_1, uav_2], True,  curr_sim_time)
            uav_1.controller, uav_2.controller = controller, controller
            print("### intermediate saved to ", exp_path)


        
        
        # advance timer and step counter:
        iteration_time += steps_per_call * time_step
        curr_sim_time += steps_per_call * time_step
        curr_step += steps_per_call

        # reset episode
        if iteration_time > ITERATION_TIME or uav_2.states[-1][1] > y_pos_max:
            print("Collected ", len(time_seq), "samples of data.")
            print("Max. force on uav 2 recorded: ", np.max(np.abs(np.array(uav_2.states)[:,8]*DRONE_TOTAL_MASS)))
            print("Max. Y-vel. of uav 2 recorded: ", np.max(np.abs(np.array(uav_2.states)[:,4])))
            print("Min height of uav 1:", np.min(np.array(uav_1.states)[:,2]))
            print("Min height of uav 2:", np.min(np.array(uav_2.states)[:,2]))
            print("-----------------------------------------")

            # init next iteration

            initial_point = np.array((0.0,0.0,0.0))#sample_3d_point(1.0) # returns at negative side a point p=[N(), N(), N()]
            sampled_y_vel = sample_from_range(y_vel_min,y_vel_max)
            initial_y_pos = -2.5#-1.5 - sampled_y_vel*0.2
            selected_velocities.append(sampled_y_vel)
            ITERATION_TIME = HOVER_DURATION + 12.0/sampled_y_vel

            px4_input_1 = (0.0, 0.0, 0.0, 0.0, nan, nan, nan, nan, nan, nan, yaw_uav_1, nan) # uav 1 hover at (0,0,0)
            px4_input_2 = (0.0, initial_point[0], initial_y_pos, initial_point[2]-2.0, nan, nan, nan, nan, nan, nan, yaw_uav_2, nan) # uav 2 hover at (x,0,z)
            iteration_time = 0.0
            total_iterations += 1
            print("### Sim time:", curr_sim_time, "/", SIM_MAX_DURATION, "s", " ## Sim steps:", curr_step ,"/", total_sim_steps, "steps ###")
            print("### Allocated time for next episode:", ITERATION_TIME)
            print("Sampled x and z positions:", initial_point[0], initial_point[2])
            print("### Sampled Velocity:", sampled_y_vel)
            print("beginning iteration at global time",np.round(curr_sim_time,2))


            controller.clear()
            controller.start()


    # Clear and close simulator

    print("Finished, clearing...")
    controller.clear()
    controller.close()
    print("Control loop ended.")
    print("Collected ", len(time_seq), "samples of data,",str(SIM_MAX_DURATION),"s, over", str(total_iterations), "iterations.")
    print("Selected velocities:", selected_velocities)

    #if SAVE_EXP:
    #    uav_1.controller, uav_2.controller = None, None
    #    exp_path = save_experiment(exp_name, [uav_1, uav_2], True,  SIM_MAX_DURATION)
    #    print("### Finally saving experiment to ", exp_path)



    extract_and_plot_data(None, None, np.array(time_seq), uav_1.states, uav_2.states,plot=True, roll_iterations=True, save=SAVE_EXP,
                          data_name=dataset_name)
    #extract_and_plot_data(None, None, np.array(time_seq), uav_1.states, uav_2.states,plot=True, roll_iterations=True)
    
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



