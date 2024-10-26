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
    SIM_DURATION = 25.0
    spin_up_time = 5.0
    throttle = 0.2

    # connect to simulators controller
    controller = simcontrol2.Controller("localhost", port)
    controller.clear()

    ext_force_sensors = ["force_sensor_body_z", "force_sensor_rotor_1_z", "force_sensor_rotor_2_z", "force_sensor_rotor_3_z", "force_sensor_rotor_4_z"]
    jft_sensors = ["jft_sensor_body", "jft_sensor_imu",  "jft_sensor_rotor1", "jft_sensor_rotor2", "jft_sensor_rotor3", "jft_sensor_rotor4"]

    # setup UAVs
    uav_2_ext_z_force_sensors = ["uav_2_" + ext_sen_name for ext_sen_name in ext_force_sensors]
    uav_2_jft_sensors = ["uav_2_" + ext_sen_name for ext_sen_name in jft_sensors]
    uav_2 = UAV("sufferer", controller, "controller2", "imu2", uav_2_ext_z_force_sensors, uav_2_jft_sensors)

    # special variables

    body_jft_sensor_idx = controller.get_sensor_info("uav_2_jft_sensor").index
    

    # Start simulation
    controller.clear()
    controller.start()
    time_step = controller.get_time_step()  # time_step = 0.0001
    sim_max_duration = SIM_DURATION # sim seconds
    total_sim_steps = sim_max_duration / time_step
    control_frequency = 750.0 # Hz
    # Calculate time steps between one control period
    steps_per_call = int(1.0 / control_frequency / time_step)
    print("One Timestep is ", time_step, "||", "Steps per call are", steps_per_call,"||", 
        "Sim dt is timestep * steps per call =", time_step * steps_per_call)


    # Initialize timer
    curr_sim_time = 0.0
    curr_step = 0

    time_seq = []
    rel_state_vector_list = []

    throttle_list = []
    rotor_1_rps_list = []
    body_jft_forces_list = []

    


    # each iteration sends control signal
    while curr_sim_time < sim_max_duration * 1.1 and throttle < 0.7:

        # log everything at current timestep  first
        
        print("## Sim time:", np.round(curr_sim_time,3), "/", sim_max_duration, "s",
            " ## Sim steps:", curr_step ,"/", total_sim_steps, "steps")
        
      

        #qx, qy, qz, qw = euler_angles_to_quaternion(0,0,0)
        px4_input_2 = (3.0, 0, 0, 0, throttle) 

        # Simulate a control period, giving actuator input and retrieving sensor output
        reply = controller.simulate(steps_per_call,  { uav_2.px4_idx: px4_input_2})
        
        

        # Check simulation result
        if reply.has_error():
            print('Simulation failed! Terminating control experiement.')
            break

        
        # read sensors of uavs
        uav_2.update(reply, curr_sim_time)

        if curr_sim_time >= spin_up_time:
            throttle = (curr_sim_time-spin_up_time)/(sim_max_duration-spin_up_time) + 0.2
            print(throttle)
            time_seq.append(curr_sim_time)
            throttle_list.append(throttle)
            body_jft_forces_list.append(reply.get_sensor_output(body_jft_sensor_idx)[2])
        
        
        #print("-------")
        #print("current throttle:", throttle_list[-1], "r1 force", rotor_1_forces_list[-1], "r1 rps:", rotor_1_rps_list[-1])
        #const_c_1 = rotor_1_forces_list[-1]/rotor_1_rps_list[-1]**2
        #print("computed C_1:", const_c_1)
        #print("throttle/totalforce correspodance:", throttle_list[-1], "to", rotor_1_forces_list[-1]*4)
        #rel_state_vector_uav_2 = np.round(np.array(uav_1.states[-1]) - np.array(uav_2.states[-1]), 2)
        #rel_state_vector_list.append(rel_state_vector_uav_2)
        

        
        
        curr_sim_time += steps_per_call * time_step
        curr_step += steps_per_call


    # Clear and close simulator
    print("Finished, clearing...")
    controller.clear()
    controller.close()
    print("Control experiment ended.")
    print("Collected ", len(rel_state_vector_list), "samples of data.")

    # compute measurements and account for bias
    #jft_bias = np.mean(body_jft_forces_list[:int(spin_up_time*control_frequency)])
    jft_bias = 29.7733538
    adjusted_jft_forces = np.array(body_jft_forces_list) - jft_bias
    print("jft-bias =", jft_bias)

    fig = plt.subplot()
    fig.plot(throttle_list, body_jft_forces_list)
    fig.set_title("throttle vs force")
    fig.set_xlabel("throttle input value")
    fig.set_ylabel("Raw measurement on JFT sensor (N)")

    plt.show()
    fig = plt.subplot()
    #fig.plot(time_seq, throttle_list)
    fig.plot(time_seq, -adjusted_jft_forces)
    fig.set_title("Force vs time")
    fig.set_xlabel("time (s)")
    fig.set_ylabel("Force")
    plt.show()
    fig = plt.subplot()
    fig.plot(throttle_list, -adjusted_jft_forces)
    fig.set_title("Force function for set throttle value")
    fig.set_xlabel("throttle")
    fig.set_ylabel("Force (N)")
    plt.show()
    #plt.legend()


    # Fit a quadratic model
    a, b, c = np.polyfit(throttle_list, -adjusted_jft_forces, 2)
    print(f"Fitted coefficients now: a = {a}, b = {b}, c = {c}")

    # Function to calculate throttle for a given thrust
    def throttle_for_thrust(thrust, a, b, c):
        discriminant = b**2 - 4 * a * (c - thrust)
        if discriminant < 0:
            raise ValueError("No real solution for this thrust value.")
        
        # Calculate both solutions
        throttle1 = (-b + math.sqrt(discriminant)) / (2 * a)
        # Choose the valid throttle value within the range [0, 1]
        
        return throttle1
        
            
            
    
    fig = plt.subplot()

    fig.plot(-adjusted_jft_forces, throttle_list)
    fig.plot(-adjusted_jft_forces, [throttle_for_thrust(thrust, a,b,c) for thrust in -adjusted_jft_forces], )
    fig.set_title("Approximated thrust for given Force (N) and actual throttle")
    fig.set_xlabel("Force (N)")
    fig.set_ylabel("throttle value")
    
    plt.show()

    fig = plt.subplot()

    fig.plot(-adjusted_jft_forces, throttle_list, label="measured")
    fig.plot(-adjusted_jft_forces, [thrust_to_throttle_p005_mrv80(thrust) for thrust in -adjusted_jft_forces], label="fitted model")
    fig.plot(-adjusted_jft_forces, [thrust_to_throttle_p005_mrv80_trimmed(thrust) for thrust in -adjusted_jft_forces], label="trimmed model")


    fig.set_title("Approximated thrust for given Force (N) and actual throttle")
    fig.set_xlabel("Force (N)")
    fig.set_ylabel("throttle value")
    
    plt.show()
    
    # Example usage: Calculate throttle needed for a thrust of 5.0
    desired_thrust = 30.0
    required_throttle = throttle_for_thrust(desired_thrust, a, b, c)
    print(f"Required throttle for {desired_thrust} N thrust: {required_throttle}")
    from scipy.optimize import fsolve
    throttle_data = throttle_list
    thrust_data = -adjusted_jft_forces

    # Fit a cubic polynomial (degree 3) to the data
    coefficients = np.polyfit(throttle_data, thrust_data, 3)
    print(coefficients)
    cubic_model = np.poly1d(coefficients)

    # Define a function to find throttle for a given thrust using fsolve
    def find_throttle_for_thrust(thrust_target):
        # Root-finding function for fsolve: cubic_model(throttle) - thrust_target = 0
        func = lambda throttle: cubic_model(throttle) - thrust_target
        # Use fsolve to find the root (throttle) that makes func = 0
        throttle_solution = fsolve(func, x0=0.5)  # x0 is the initial guess
        return throttle_solution[0]

    # Example usage: find throttle for a specific thrust value
    thrust_target = 7.0
    throttle_required = find_throttle_for_thrust(thrust_target)
    print(f"Required throttle for thrust {thrust_target}: {throttle_required}")

    # Plotting
    # Use the model to predict thrust for a range of throttle values
    throttle_range = np.linspace(0, 1, 100)
    predicted_thrust = cubic_model(throttle_range)

    # Plot the cubic model (throttle to thrust)
    plt.plot(throttle_data, thrust_data, 'o', label='Data Points')       # Original data points
    plt.plot(throttle_range, predicted_thrust, '-', label='Cubic Fit')  # Cubic fit curve
    plt.xlabel("Throttle")
    plt.ylabel("Thrust")
    plt.title("Throttle vs. Thrust (Cubic Fit)")
    plt.legend()

    # Show where the target thrust maps to throttle
    plt.plot(throttle_required, thrust_target, 'ro', label=f'Target Thrust: {thrust_target}')
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


