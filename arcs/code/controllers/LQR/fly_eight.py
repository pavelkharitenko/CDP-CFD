from simcontrol import simcontrol2
import numpy as np
import matplotlib.pyplot as plt
import sys, os

sys.path.append('../../utils/')
sys.path.append('../../uav/')

from uav import *
from utils import *





from trajectory_generator import Trajectory



linearization_params = {
        'K_roll': -.00,
        'K_pitch': .005,
        'K_yaw': 0.005,
        'K_thrust': 1,  # Thrust to maintain altitude
        'x_target': 0.0,
        'y_target': 0.0,
        'z_target': 1.0,  # Target altitude
        'yaw_target': 0.0  # Target yaw angle
    }

trajectory_gen = Trajectory(linearization_parameters=linearization_params)
num_points = 100  # Number of points in the trajectory
height = 1.0      # Constant height for the horizontal eight
reference_states, x, y, z, yaw = trajectory_gen.generate_horizontal_eight_trajectory(num_points, height)
forward_velocity = 1.0  # Forward velocity in m/s
reference_states, x_forward, y_forward, z_forward, yaw_forward = trajectory_gen.generate_forward_flight_trajectory(num_points, forward_velocity, height)



port = 25556
SIM_DURATION = 3
HOVER_TIME = 1.2
FLY_CIRCULAR = False
freq = 0.02
radius = 3.0
y_velocity = 2

# connect to simulators controller
controller = simcontrol2.Controller("localhost", port)

ext_force_sensors = ["force_sensor_body_z", "force_sensor_rotor_1_z", "force_sensor_rotor_2_z", "force_sensor_rotor_3_z", "force_sensor_rotor_4_z"]
jft_sensors = ["jft_sensor_body", "jft_sensor_imu",  "jft_sensor_rotor1", "jft_sensor_rotor2", "jft_sensor_rotor3", "jft_sensor_rotor4"]


# setup UAVs
uav_1_ext_z_force_sensors = ["uav_1_" + ext_sen_name for ext_sen_name in ext_force_sensors]
uav_2_ext_z_force_sensors = ["uav_2_" + ext_sen_name for ext_sen_name in ext_force_sensors]
uav_1_jft_sensors = ["uav_1_" + ext_sen_name for ext_sen_name in jft_sensors]
uav_2_jft_sensors = ["uav_2_" + ext_sen_name for ext_sen_name in jft_sensors]

uav_1 = UAV("producer", controller, "controller1", "imu1", uav_1_ext_z_force_sensors, uav_1_jft_sensors)
uav_2 = UAV("sufferer", controller, "controller2", "imu2", uav_2_ext_z_force_sensors, uav_2_jft_sensors)

# set experiment 
nan = float('NaN')

# Start simulation
controller.start()
time_step = controller.get_time_step()  # time_step = 0.0001
sim_max_duration = SIM_DURATION # sim seconds
total_sim_steps = sim_max_duration / time_step
control_frequency = 30.0 # Hz
# Calculate time steps between one control period
steps_per_call = int(1.0 / control_frequency / time_step)
print("One Timestep is ", time_step, "||", "Steps per call are", steps_per_call,"||", 
      "Sim dt is timestep * steps per call =", time_step * steps_per_call)


# Initialize timer
curr_sim_time = 0.0
curr_step = 0

time_seq = []
rel_state_vector_list = []

current_idx = 0

current_position = (0,0,1)
target_vel = [0,0,0]

def waypoints_to_velocities(current_position, waypoints, dt):
    if not waypoints:
        return np.zeros(3)  # No waypoints, return zero velocity
    
    # Get the next waypoint
    desired_position = waypoints[0]
    
    # Calculate the velocity vector
    velocity = (np.array(desired_position) - np.array(current_position)) / dt
    
    # Optionally, limit the velocity to a maximum value
    max_velocity = 1.0  # Define max velocity
    velocity_magnitude = np.linalg.norm(velocity)
    if velocity_magnitude > max_velocity:
        velocity = (velocity / velocity_magnitude) * max_velocity
    
    return velocity




# each iteration sends control signal
while curr_sim_time < sim_max_duration:

    # log everything at current timestep  first
    time_seq.append(curr_sim_time)
    print("## Sim time:", np.round(curr_sim_time,3), "/", sim_max_duration, "s",
          " ## Sim steps:", curr_step ,"/", total_sim_steps, "steps")
    

    
    if curr_sim_time < HOVER_TIME:
        #px4_input_1 = (0.0, nan,nan,nan, 0.0, 0.0, 0.0, nan, nan, nan, 0.0, nan) 
        px4_input_1 = (0.0, 0.0, -1, 0.6, nan, nan, nan, nan, nan, nan, 0.0, nan) 
    else:
        # keep position
        px4_input_1 = (0.0, nan, nan, 0.6, 0.0, y_velocity, nan, nan, nan, nan, 0.0, nan) 
        #px4_input_1 = (0.0, nan,nan,nan, 0.0, 0.0, 0.0, nan, nan, nan, 0.0, nan) 
            
    # second drone controlled by torques and thrust
    next_waypoint = (1,0,1)




    
    px4_input_2 = (2.0, 0.01*target_vel[0], 0.01*target_vel[1], 0.01*target_vel[2], 0.5)

    # Simulate a control period, giving actuator input and retrieving sensor output
    reply = controller.simulate(steps_per_call,  {
                                                   uav_1.px4_idx: px4_input_1, 
                                                   uav_2.px4_idx: px4_input_2, 
                                                  })
    
    print(target_vel)

    # Check simulation result
    if reply.has_error():
        print('Simulation failed! Terminating control experiement.')
        break

    
    # read sensors of uavs
    uav_1.update(reply, curr_sim_time)
    uav_2.update(reply, curr_sim_time)

    current_position = uav_2.states[-1][:3]
    target_vel = waypoints_to_velocities(current_position, next_waypoint, steps_per_call * time_step)
    
    rel_state_vector_uav_2 = np.round(np.array(uav_1.states[-1]) - np.array(uav_2.states[-1]), 2)
    #print(rel_state_vector)
    rel_state_vector_list.append(rel_state_vector_uav_2)
    

    # advance timer and step counter:
    curr_sim_time += steps_per_call * time_step
    curr_step += steps_per_call
    current_idx += 1


# Clear and close simulator
print("Finished, clearing...")
controller.clear()
controller.close()
print("Control experiment ended.")

print("Collected ", len(rel_state_vector_list), "samples of data.")


model_paths = [#r"C:\Users\admin\Desktop\IDP\CDP-CFD\arcs\code\observers\ndp\2024-10-09-12-25-27-NDP-Li-Model-sn_scale-None-144k-datapoints-corrected-bias-complicated-tropics20000_eps.pth",
               #r"C:\Users\admin\Desktop\IDP\CDP-CFD\arcs\code\observers\ndp\2024-10-09-21-24-10-NDP-Li-Model-sn_scale-4-144k-datapoints-corrected-bias-sizzling-speed20000_eps.pth",
               #r"C:\Users\admin\Desktop\IDP\CDP-CFD\arcs\code\observers\ndp\2024-10-10-10-56-56-NDP-Li-Model-sn_scale-2-144k-datapoints-corrected-bias-edible-status20000_eps.pth",
               #r"C:\Users\admin\Desktop\IDP\CDP-CFD\arcs\code\observers\ndp\2024-10-09-10-25-10-NDP-Li-Model-sn_scale-None-114k-datapoints-inventive-defilade20000_eps.pth",
               #r"C:\Users\admin\Desktop\IDP\CDP-CFD\arcs\code\observers\ndp\2024-10-09-10-47-22-NDP-Li-Model-sn_scale-4-114k-datapoints-stubborn-content20000_eps.pth"
               ]

models = []
for model_path in model_paths:
    model = DWPredictor().to(device)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    models.append(model)

predictions = evaluate_zy_force_curvature(models, rel_state_vector_list)
labels = ["Li's NDP (no sn)","sn<=4", "sn <=2", "no sn, old bias","sn<4 old bias"]









for idx, prediction in enumerate(predictions):
    print("plotting pred.")
    plt.plot(time_seq, prediction, label=labels[idx])


plt.xlabel("time (s)")
plt.ylabel("Force (N)")
plt.title(f"Measured and predicted DW forces at\nSpeed of Y={y_velocity} (Both Hovering)")




Fz_total = [uav_2.total_mass * state[8] for state in uav_2.states] # recorded actual uav m*a z-Force
Fg = [uav_2.total_mass * 9.81 for state in uav_2.states] # uav gravitational force m*g
Fz_u_total = np.array([-np.sum(body_r1_r2_r3_r4, axis=0) for body_r1_r2_r3_r4 in uav_2.jft_forces_list])
g_bias_steady_state = 4.5 # on average, Fu compentates always more by 5N, so it is not disturbance force

dw_forces = np.array(Fz_total) + np.array(Fg) - np.array(Fz_u_total)
dw_forces += g_bias_steady_state
#print("dw forces raw:", dw_forces[-500:-400])
#print("dw forces after subtracting bias:", dw_forces[-500:-400])

dw_force_vectors = np.array([(0,0,dw_force) for dw_force in dw_forces])


plt.plot(time_seq,[meas[2] for meas in dw_force_vectors], label='Recorded z-force')
plt.legend()
plt.show()


