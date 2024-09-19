from simcontrol import simcontrol2
import numpy as np

# Create simulation controller
controller = simcontrol2.Controller('localhost', 25556)

# Retrieve px4 actuator index (px4 is a class of actuator)
px4_index = controller.get_actuator_info('controller').index

# Retrieve imu sensor index (imu is a class of sensor)
imu_index = controller.get_sensor_info('imu').index

# Start simulation
controller.start()

# Retrive simulation time step (this should be after simulation start)
time_step = controller.get_time_step()

# Calculate time steps between one control period
control_frequency = 100.0
steps_per_call = int(1.0 / control_frequency / time_step)

# Initialize timer
t = 0.0

# Simulation loop, simulating for 20 seconds (simulation time, not physical time)
while (t < 20.0):
    # Set px4 control input to do position tracking of a circle trajectory
    # See actuator.md for details of input
    freq = 0.25
    radius = 1.0
    px = radius * np.sin(2.0 * np.pi * freq * t)
    py = radius - radius * np.cos(2.0 * np.pi * freq * t)
    nan = float('NaN')
    px4_input = (0.0, px, py, 1.0, nan, nan, nan, nan, nan, nan, 0.0, nan) # This is the actuator input vector

    # Simulate a control period, giving actuator input and retrieving sensor output
    reply = controller.simulate(steps_per_call,  {px4_index: px4_input})

    # Advance timer
    t += steps_per_call * time_step

    # Check simulation result
    if reply.has_error():
        print('Simulation failed!')
        break
    else:
        # Get imu sensor output (a tuple of floats)
        imu_data = reply.get_sensor_output(imu_index)

        # Extract copter position from imu readings
        # See sensor.md for details of output
        p_x = imu_data[0]
        p_y = imu_data[1]
        p_z = imu_data[2]

        # Logging or data processing...
        print(f'x: {p_x:.2f}, y: {p_y:.2f}, z: {p_z:.2f}')

# Clear simulator
print("Finished, clearing...")
controller.clear()

# Close the simulation controller
controller.close()
print("Done.")
