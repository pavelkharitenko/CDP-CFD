from simcontrol import simcontrol2
import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.append('../../observers/baseline/')
sys.path.append('../../uav/')
sys.path.append('../../utils/')
from uav import *
from utils import *

def main(controller):
    port = 25556
    SIM_DURATION = 15.0
    spin_up_time = 3.0
    measurement_duration = 10.0 # sim seconds
    rotor_ds = 1.0
    rotor_measurements = np.round(np.arange(401, 401 + rotor_ds, rotor_ds), 1)
    #throttle_measurements = [0.5]

    # connect to simulators controller
    controller = simcontrol2.Controller("localhost", port)
    controller.clear()

    

    # special variables

    body_jft_sensor_idx = controller.get_sensor_info("uav_2_jft_sensor").index
    r1_joint_sensor_idx = controller.get_sensor_info("uav_2_r1_joint_sensor").index
    r2_joint_sensor_idx = controller.get_sensor_info("uav_2_r2_joint_sensor").index
    r3_joint_sensor_idx = controller.get_sensor_info("uav_2_r3_joint_sensor").index
    r4_joint_sensor_idx = controller.get_sensor_info("uav_2_r4_joint_sensor").index

    uav_2_r1_idx = controller.get_actuator_info("uav_2_r1_joint_motor").index
    uav_2_r2_idx = controller.get_actuator_info("uav_2_r2_joint_motor").index
    uav_2_r3_idx = controller.get_actuator_info("uav_2_r3_joint_motor").index
    uav_2_r4_idx = controller.get_actuator_info("uav_2_r4_joint_motor").index


    

    # Start simulation
    controller.clear()
    controller.start()
    time_step = controller.get_time_step()  # time_step = 0.0001
    control_frequency = 1000.0 # Hz
    # Calculate time steps between one control period
    steps_per_call = int(1.0 / control_frequency / time_step)


    # Initialize timer
    curr_sim_time = 0.0
    curr_step = 0

    


    raw_body_jft_forces_list = []
    rotor_1_rps_list = []
    rotor_2_rps_list = []
    rotor_3_rps_list = []
    rotor_4_rps_list = []


    # each iteration sends control signal
    for rotor_meas in rotor_measurements:

        print(f"Measuring rps {rotor_meas}/{rotor_measurements[-1]}")
        curr_sim_time = 0
        curr_step = 0

        current_rotor_measurements = []
        current_r1_measurements = []
        current_r2_measurements = []
        current_r3_measurements = []
        current_r4_measurements = []


        while curr_sim_time <= measurement_duration + spin_up_time:
        
             

            
            reply = controller.simulate(steps_per_call,  { 
                                                    uav_2_r1_idx: (rotor_meas,), 
                                                   uav_2_r2_idx: (rotor_meas,), 
                                                   uav_2_r3_idx: (-rotor_meas,),  # front right
                                                   uav_2_r4_idx: (-rotor_meas,),  # back left})
            })
            
            
            if curr_sim_time >= spin_up_time:
                current_rotor_measurements.append(-reply.get_sensor_output(body_jft_sensor_idx)[2])
                current_r1_measurements.append(reply.get_sensor_output(r1_joint_sensor_idx)[0])
                current_r2_measurements.append(reply.get_sensor_output(r2_joint_sensor_idx)[0])
                current_r3_measurements.append(reply.get_sensor_output(r3_joint_sensor_idx)[0])
                current_r4_measurements.append(reply.get_sensor_output(r4_joint_sensor_idx)[0])



            # advance simulation
            curr_sim_time += steps_per_call * time_step
            curr_step += steps_per_call
            #print(curr_sim_time)

        raw_body_jft_forces_list.append(current_rotor_measurements)
        print("avg. force recorded:", np.round(np.mean(raw_body_jft_forces_list[-1]),2))
        rotor_1_rps_list.append(current_r1_measurements)
        rotor_2_rps_list.append(current_r2_measurements)
        rotor_3_rps_list.append(current_r3_measurements)
        rotor_4_rps_list.append(current_r4_measurements)
        print("predicted avg:", rps_to_thrust_p005_mrv80(np.mean(current_r1_measurements)) - 29.7733538)


        
        controller.clear()
        controller.start()



    # Clear and close simulator
    print("Finished, clearing...")

    #jft_bias = 29.7733538

    for idx, force_meas in enumerate(raw_body_jft_forces_list):
        
        if idx % 5 == 0:
            adjusted_jft_forces = np.array(force_meas)    # - jft_bias
            fig = plt.subplot()
            fig.plot(adjusted_jft_forces, label=f"{rotor_measurements[idx]}")
            fig.plot(np.full(len(adjusted_jft_forces), np.mean(adjusted_jft_forces)), label=f"{rotor_measurements[idx]} - mean")


    plt.legend()
    plt.show()
    
    
    
    force_averages = [np.mean(sublist[int(spin_up_time * control_frequency):]) for sublist in raw_body_jft_forces_list]
    r1_averages = [np.mean(sublist[int(spin_up_time * control_frequency):]) for sublist in rotor_1_rps_list]
    r2_averages = [np.mean(sublist[int(spin_up_time * control_frequency):]) for sublist in rotor_2_rps_list]
    r3_averages = [np.mean(sublist[int(spin_up_time * control_frequency):]) for sublist in rotor_3_rps_list]
    r4_averages = [np.mean(sublist[int(spin_up_time * control_frequency):]) for sublist in rotor_4_rps_list]

    rotors_avgs = np.mean([np.abs(r1_averages), np.abs(r2_averages), np.abs(r3_averages), np.abs(r4_averages)], axis=0)



    # Function to create a beautiful table plot with matplotlib
    def plot_table(data):
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.axis('tight')
        ax.axis('off')
        
        # Create a table from the data
        table_data = [["Rps value", 
                       f"Average Force (N) over {measurement_duration}s\n(after some spinup time)", 
                       "r1 avg. rps", "r2 avg. rps","r3 avg. rps","r4 avg. rps",
                       "mean rotor avg rps"
                       ]]
        for i, avg in enumerate(data):
            table_data.append([rotor_measurements[i], 
                               f"{avg:.4f}", 
                               f"{r1_averages[i]:.4f}", f"{r2_averages[i]:.4f}",f"{r3_averages[i]:.4f}",f"{r4_averages[i]:.4f}",
                               f"{rotors_avgs[i]:.4f}"])
        
        # Display table
        table = ax.table(cellText=table_data, colLabels=None, cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.0, 1.0)

        # Set table style
        for i in range(len(data) + 1):
            for j in range(2):
                table[(i, j)].set_edgecolor("black")
        plt.show()

    # Plot the table
    plot_table(force_averages)

    plt.scatter(rotor_measurements, force_averages)
    plt.show()
    #np.savez("rotor_thrust_map_005_80_rps361_400_430.npz", rotor_measurements=rotor_measurements, force_averages=force_averages)
    






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


