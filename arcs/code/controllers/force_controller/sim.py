from simcontrol import simcontrol2
from uav import UAV
from code.controllers.force_controller.force_controller import LQRController
import numpy as np
np.set_printoptions(precision=2)
np.set_printoptions(suppress=True)


# Init

# connect to simulators controller
ar_controller = simcontrol2.Controller("localhost", 25556)
nan = float('NaN')

jft_sensors = ["jft_sensor_rotor1", "jft_sensor_rotor2", "jft_sensor_rotor3", "jft_sensor_rotor4"]

# setup UAVs
uav_2_jft_sensors = ["uav_2_" + ext_sen_name for ext_sen_name in jft_sensors]
uav = UAV("sufferer", ar_controller, "controller2", "imu2", uav_2_jft_sensors,  dt=0.00125)


# Define cost matrices for LQR
Q = np.eye(7)  # Penalize state deviations
R = np.eye(4)  # Penalize control effort

# Initialize LQR controller
lqr_controller = LQRController(uav.A, uav.B, Q, R)


# Run Simulation

def run_simulation():

    # Start simulation
    ar_controller.start()
    time_step = ar_controller.get_time_step()  # time_step = 0.0001
    sim_max_duration = 10 # sim seconds
    total_sim_steps = sim_max_duration / time_step
    control_frequency = 30.0 # Hz
    # Calculate time steps between one control period
    steps_per_call = int(1.0 / control_frequency / time_step)
    print("One Timestep is ", time_step, "||", "Steps per call are", steps_per_call,"||",  "Sim dt is timestep * steps per call =", time_step * steps_per_call)
    
    # Initialize timer
    curr_sim_time = 0.0
    curr_step = 0

    time_seq = []
    px4_commands = (0.0, 
                        nan,nan,nan, 
                        nan,nan,nan, 
                        0.0, 0.0, 0.0,
                        nan, 0.0)


    while curr_sim_time < sim_max_duration:

        time_seq.append(curr_sim_time)
        print("## Sim time:", np.round(curr_sim_time,3), "/", sim_max_duration, "s", " ## Sim steps:", curr_step ,"/", total_sim_steps, "steps")


        reply = ar_controller.simulate(steps_per_call,  {uav.px4_idx: px4_commands})
        
        # Step 1: Read IMU data (position, velocity, yaw) from the actual UAV
        uav.update(reply, curr_sim_time)

        # Step 3: Get the current state of the UAV from the model
        current_state = uav.output()

        # Step 4: Compute control input using the LQR controller
        control_input = lqr_controller.control_input(current_state)

        #print("Control input:", control_input)

        # Step 5: Pass control inputs to PX4 controller
        ax, ay, az, yaw_rate = control_input

        px4_commands = (0.0, 
                        nan,nan,nan,                    
                        nan, nan, nan, 
                        0.0,0.0, 0.0,
                        0.0, 0.0)
        

                        #0.01*ax, 0.01*ay, 0.01*az,
        # Step 6: Optionally update the UAV model with the control input
        #new_state = uav.state_update(control_input)

        # Debug or log the state and control inputs
        print(f"Current state= {np.array(current_state)}, Control Input = {np.array(control_input)}, PX4 Commands = {np.array(px4_commands)}")

        # advance timer and step counter:
        curr_sim_time += steps_per_call * time_step
        curr_step += steps_per_call



run_simulation()