from simcontrol import simcontrol2

# connect to simulators controller
port = 25556
controller = simcontrol2.Controller("localhost", port)

# retrieve px4 actuator index:
px4_index_1 = controller.get_actuator_info('controller1').index

# retrieve px4 actuator index:
px4_index_2 = controller.get_actuator_info('controller2').index

# setup IMUs
# Retrieve imu sensor index (imu is a class of sensor)
imu1_index = controller.get_sensor_info('imu1').index

imu2_index = controller.get_sensor_info('imu2').index

# set experiment 
nan = float('NaN')

# Start simulation
controller.start()
time_step = controller.get_time_step()  # time_step = 0.0001
sim_max_duration = 30.0 # sim seconds
total_sim_steps = sim_max_duration / time_step
control_frequency = 30.0 # Hz
# Calculate time steps between one control period
steps_per_call = int(1.0 / control_frequency / time_step)
print(
    "One Timestep is ", time_step, "||",
    "Steps per call are", steps_per_call,"||", 
    "Sim dt is timestep * steps per call =", time_step * steps_per_call)


# Initialize timer
curr_sim_time = 0.0
curr_step = 0

# each iteration sends control signal
while curr_sim_time < sim_max_duration:

    # log everything at current timestep  first
    print("## Sim time:", curr_sim_time, "/", sim_max_duration, "s",
          " ## Sim steps:", curr_step ,"/", total_sim_steps, "steps")
    
    # Create controller input. This is the actuator input vector

    # uav 2 flies to right after 2.6 seconds
    #px4_input_1 = (0.0, 0.0, 0.0, 2.0, nan, nan, nan, nan, nan, nan, 0.0, nan) 
    #if curr_sim_time < 2.6:
    #    px4_input_2 = (0.0, 0.0, -1.5, 1.5, nan, nan, nan, nan, nan, nan, 0.0, nan) 
    #else:
    #    px4_input_2 = (0.0, 0.0, 1.5, 1.5, nan, nan, nan, nan, nan, nan, 0.0, nan) 

    px4_input_2 = (0.0, 0.0, -1.5, 1.5, nan, nan, nan, nan, nan, nan, 0.0, nan) 

    # uav 1  flies over left
    if curr_sim_time < 3.6:
        px4_input_1 = (0.0, 0.0, 0.0, 2.0, nan, nan, nan, nan, nan, nan, 0.0, nan) 
    else:
        px4_input_1 = (0.0, 0.0, -1.5, 2.0, nan, nan, nan, nan, nan, nan, 0.0, nan) 

    # Simulate a control period, giving actuator input and retrieving sensor output
    reply = controller.simulate(steps_per_call,  {px4_index_1: px4_input_1, 
                                                  px4_index_2: px4_input_2})


    


    # advance timer and step counter:
    curr_sim_time += steps_per_call * time_step
    curr_step += steps_per_call


# Clear simulator
print("Finished, clearing...")
controller.clear()

# Close the simulation controller
controller.close()
print("Done.")