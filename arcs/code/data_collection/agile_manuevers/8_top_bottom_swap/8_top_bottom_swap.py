from simcontrol import simcontrol2
import numpy as np
import matplotlib.pyplot as plt
import sys, torch

sys.path.append('../../../uav/')
sys.path.append('../../../utils/')
sys.path.append('../../../controllers/nonlinear_feedback')
sys.path.append('../../../observers/')

from agile.model import AgileShallowPredictor
from ndp.model import DWPredictor
from controller3 import NonlinearFeedbackController3
from planner import Planner

from scipy.spatial.transform import Rotation as R


from uav import *
from utils import *

def main(controller):
    # manuever specific
    MANUEVER_NAME = "1_nfc_flybelow"
    port = 25556
    nan = float('NaN')

    # exp specific
    SIM_MAX_DURATION = 15.0
    HOVER_DURATION = 0.0
    
    
    total_iterations = 0
    selected_velocities = []

    # controller & uav init
    controller = simcontrol2.Controller("localhost", port)
    uav_1, uav_2 = init_two_uavs(controller)

    nfc1 = NonlinearFeedbackController3()
    nfc2 = NonlinearFeedbackController3()
    yaw_uav_1 = 1.570796326794897
    yaw_uav_2 = 1.570796326794897 #* 180.0/np.pi

    
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
    curr_step = 0

    target_velocity = 0.5
    acceleration_time = 10.0
    hover_time = 3.0

    planner1 = Planner(velocity=target_velocity, acceleration_time=acceleration_time, dt=0.005, hover_time=hover_time, 
                      start=(-3.0,0,0), 
                      end=(0.0,7.0,-1.5), 
                      initial_yaw=yaw_uav_1, 
                      traj_type=4)
    planner1.plot_trajectory_2d()
    planner1.plot_trajectory()

    planner2 = Planner(velocity=target_velocity, acceleration_time=acceleration_time, dt=0.005, hover_time=hover_time, 
                      start=(0.0,0,0), 
                      end=(0.0,7.0,-1.5), 
                      initial_yaw=yaw_uav_2, 
                      traj_type=3)
    planner2.plot_trajectory_2d()
    planner2.plot_trajectory()

    #exit(0)
    current_waypoint1 = planner1.pop_waypoint(np.zeros(9), alpha=1.0)
    current_waypoint2 = planner2.pop_waypoint(np.zeros(9), alpha=1.0)

    # load disturbance predictor

    predictor1 = AgileShallowPredictor()
    predictor1.load_state_dict(torch.load(find_file_with_substring("upbeat-elk30"), weights_only=True))
    predictor2 = AgileShallowPredictor()
    predictor2.load_state_dict(torch.load(find_file_with_substring("upbeat-elk30"), weights_only=True))

    ndp_predictor1 = DWPredictor()
    ndp_predictor1.load_state_dict(torch.load(find_file_with_substring("adaptive-partition20"), weights_only=True))
    ndp_predictor2 = DWPredictor()
    ndp_predictor2.load_state_dict(torch.load(find_file_with_substring("adaptive-partition20"), weights_only=True))


    time_seq = []
    rel_state_vector_list = []
    positions1 = []
    velocities1 = []
    accelerations1 = []
    planned_pos1 = []
    positions2 = []
    velocities2 = []
    accelerations2 = []
    planned_pos2 = []

    feedforward2_forces_z = []
    uav2_actual_forces = []
    uav2_thrust_forces = []
    nf2_actual_forces = []
    
    uav2_xyz_force = np.zeros(3)
    feedforward2 = np.zeros(3)
    ndp_feedforward2 = np.zeros(3)

    feedforward1_forces_z = []
    uav1_actual_forces = []
    uav1_thrust_forces = []
    nf1_actual_forces = []
    
    uav1_xyz_force = np.zeros(3)
    feedforward1 = np.zeros(3)
    ndp_feedforward1 = np.zeros(3)


    #px4_input_1 = (0.0,-1.0, 0.0, 1.0, nan, nan, nan, nan, nan, nan, yaw_uav_1, nan) # uav 1 hover at (0,0,0)
    


    # each iteration sends control signal
    while curr_sim_time < SIM_MAX_DURATION:

        if curr_sim_time < HOVER_DURATION:
            px4_input_1 = (0.0, 0.0, 0.0, 0.0, nan, nan, nan, nan, nan, nan, yaw_uav_1, nan)
            px4_input_2 = (0.0, 0.0, -2.5, -1.5, nan, nan, nan, nan, nan, nan, yaw_uav_2, nan)
            reply = controller.simulate(steps_per_call,  { uav_1.px4_idx: px4_input_1, uav_2.px4_idx: px4_input_2 })
            curr_sim_time += steps_per_call * time_step
            curr_step += steps_per_call
            continue

        print("-------------")


        desired_waypoint1, desired_yaw1 = current_waypoint1[:9], current_waypoint1[9]
        nfc1.target_yaw = desired_yaw1
        desired_waypoint2, desired_yaw2 = current_waypoint2[:9], current_waypoint2[9]
        nfc2.target_yaw = desired_yaw2

        
        #feedforward = ndp_feedforward # enable alternative predictor
        feedforward1_forces_z.append(feedforward1[2])
        uav1_thrust_forces.append(uav1_xyz_force)
        feedforward2_forces_z.append(feedforward2[2])
        uav2_thrust_forces.append(uav2_xyz_force)
        #feedforward1 = np.zeros(3) # disable feedforward1 term
        #feedforward2 = np.zeros(3) # disable feedforward2 term

        px4_input_1 = nfc1.nonlinear_feedback_qt_px4(desired_waypoint1, feedforward1)     
        px4_input_2 = nfc2.nonlinear_feedback_qt_px4(desired_waypoint2, feedforward2)

        # clear feedforward term again
        feedforward1 = np.zeros(3)
        ndp_feedforward1 = np.zeros(3)
        feedforward2 = np.zeros(3)
        ndp_feedforward2 = np.zeros(3)



    
        reply = controller.simulate(steps_per_call,  { uav_1.px4_idx: px4_input_1, uav_2.px4_idx: px4_input_2 })
        
        # Check simulation result
        if reply.has_error():
            print('Simulation failed! Terminating control experiement.')
            break
        
        time_seq.append(curr_sim_time)
        uav_1.update(reply, curr_sim_time)
        uav_2.update(reply, curr_sim_time)

        pos2 = uav_2.states[-1][:3]; vel2 = uav_2.states[-1][3:6]; acc2 = uav_2.states[-1][6:9]
        pos1 = uav_1.states[-1][:3]; vel1 = uav_1.states[-1][3:6]; acc1 = uav_1.states[-1][6:9]

        # for debug, report uav2 acual and controller forces in z-axis
        uav2_actual_forces.append(np.array(acc2[:3]) * uav_2.total_mass)

        uav2_xyz_force = get_thrust_xyz_forces(np.array(uav_2.states[-1]))
        nf2_actual_forces.append(nfc2.fxyz)

        nfc1.feedback(pos1, vel1, acc1)
        nfc2.feedback(pos2, vel2, acc2)
        positions1.append(pos1); velocities1.append(vel1); planned_pos1.append(current_waypoint1)
        positions2.append(pos2); velocities2.append(vel2); planned_pos2.append(current_waypoint2)


        # feedforward1
        if np.linalg.norm(np.array(pos1[:2]) - np.array(pos2[:2]))<1.2:
            #nfc2.s_int = np.zeros(3)
            #uav_1.states[-1][0] += 2.0
            feedforward1 = predictor1.evaluate(np.array(uav_2.states[-1]).reshape(1,-1), np.array(uav_1.states[-1]).reshape(1,-1))[0]

            #feedforward = predictor.evaluate(np.array(uav_1.states[-1]).reshape(1,-1), np.array(uav_2.states[-1]).reshape(1,-1))[0]
            ndp_feedforward1 = ndp_predictor1.evaluate(np.array(uav_1.states[-1])[:6] - np.array(uav_2.states[-1])[:6])

            feedforward1 *= 0.8
            #feedforward -= np.array([0.0,-5.0,0.0]) # add y error


        # feedforward2
        if np.linalg.norm(np.array(pos2[:2]) - np.array(pos1[:2]))<1.2:
            #nfc2.s_int = np.zeros(3)
            #uav_1.states[-1][0] += 2.0
            feedforward2 = predictor2.evaluate(np.array(uav_1.states[-1]).reshape(1,-1), np.array(uav_2.states[-1]).reshape(1,-1))[0]

            #feedforward = predictor.evaluate(np.array(uav_1.states[-1]).reshape(1,-1), np.array(uav_2.states[-1]).reshape(1,-1))[0]
            ndp_feedforward2 = ndp_predictor2.evaluate(np.array(uav_2.states[-1])[:6] - np.array(uav_1.states[-1])[:6])

            feedforward2 *= 0.8
            #feedforward -= np.array([0.0,-5.0,0.0]) # add y error
            

            


        # check if current target already reached (Planner 1)
        if np.linalg.norm(np.array(pos1[:2]) - current_waypoint1[:2])<0.35:
            current_waypoint1 = planner1.pop_waypoint(uav_1.states[-1][:9])

        if np.linalg.norm(np.array(pos2[:2]) - current_waypoint2[:2])<0.35:
            current_waypoint2 = planner2.pop_waypoint(uav_2.states[-1][:9])


        rel_state_vector_uav_2 = np.round(np.array(uav_1.states[-1]) - np.array(uav_2.states[-1]), 3)
        rel_state_vector_list.append(rel_state_vector_uav_2)
        

        # advance timer and step counter:
        print("t:", np.round(curr_sim_time,3))
        curr_sim_time += steps_per_call * time_step
        curr_step += steps_per_call



    # Clear and close simulator
    print("Finished, clearing...")
    controller.clear()
    controller.close()
    print("Control loop ended.")
    print("Collected ", len(time_seq), "samples of data,",str(SIM_MAX_DURATION),"s, over", str(total_iterations), "iterations.")
    print("Selected velocities:", selected_velocities)

    ignore_start = int(hover_time*control_frequency)

    plot_trajectory_analysis_two_uavs(np.array(positions1), np.array(planned_pos1), np.array(velocities1), 
                                      np.array(positions2), np.array(planned_pos2), np.array(velocities2), 
                                      feedforward=np.array(feedforward2_forces_z),ignore_start=ignore_start)
    
    plot_trajectory_analysis_rmse(np.array(positions1), np.array(planned_pos1), np.array(velocities1), 
                                      np.array(positions2), np.array(planned_pos2), np.array(velocities2),
                                      ignore_start=ignore_start)
    
    plot_uav_positions_and_errors(np.array(positions1),np.array(positions2), target_positions1=np.array(planned_pos1)[:,:3],
                                  target_positions2=np.array(planned_pos2)[:,:3], dt=nfc2.dt)
    
    analyze_forces(uav_forces=uav2_actual_forces, thrust_forces=uav2_thrust_forces, 
                   predictor_forces_z=feedforward2_forces_z, nfc_forces=nf2_actual_forces,
                   dt=nfc2.dt)

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



