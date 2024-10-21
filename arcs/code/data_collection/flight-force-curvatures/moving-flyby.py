from simcontrol import simcontrol2
import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.append('../../observers/')
sys.path.append('../../uav/')
sys.path.append('../../utils/')

from SO2.model import ShallowEquivariantPredictor
from ndp.model import DWPredictor
from neuralswarm.model import NeuralSwarmPredictor

from uav import *
from utils import *

port = 25556
SIM_DURATION = 5.0
DRONE_TOTAL_MASS = 3.035 # P600 weight
HOVER_TIME = 1.2
FLY_CIRCULAR = False
freq = 0.02
radius = 3.0
y_velocity = 0.75

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
control_frequency = 1000.0 # Hz
# Calculate time steps between one control period
steps_per_call = int(1.0 / control_frequency / time_step)
print("One Timestep is ", time_step, "||", "Steps per call are", steps_per_call,"||", 
      "Sim dt is timestep * steps per call =", time_step * steps_per_call)



# Initialize timer
curr_sim_time = 0.0
curr_step = 0

time_seq = []
rel_state_vector_list = []


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
            
    # second drone just hovering
    px4_input_2 = (0.0, 0.0, 0.0, 0.0, nan, nan, nan, nan, nan, nan, 0.0, nan)

    # Simulate a control period, giving actuator input and retrieving sensor output
    reply = controller.simulate(steps_per_call,  {
                                                   uav_1.px4_idx: px4_input_1, 
                                                   uav_2.px4_idx: px4_input_2, 
                                                  })
    
    

    # Check simulation result
    if reply.has_error():
        print('Simulation failed! Terminating control experiement.')
        break

    
    # read sensors of uavs
    uav_1.update(reply, curr_sim_time)
    uav_2.update(reply, curr_sim_time)
    
    rel_state_vector_uav_2 = np.round(np.array(uav_1.states[-1]) - np.array(uav_2.states[-1]), 2)
    
    rel_state_vector_list.append(rel_state_vector_uav_2[:6])
    

    # advance timer and step counter:
    curr_sim_time += steps_per_call * time_step
    curr_step += steps_per_call


# Clear and close simulator
print("Finished, clearing...")
controller.clear()
controller.close()
print("Control experiment ended.")

print("Collected ", len(rel_state_vector_list), "samples of data.")


# Plot recorded and predicted forces

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

model_paths = [
    r"C:\Users\admin\Desktop\IDP\CDP-CFD\arcs\code\observers\ndp\trained_models\100Hz-data-both-drones-summed-dataset\2024-10-09-21-24-10-NDP-Li-Model-sn_scale-4-144k-datapoints-corrected-bias-sizzling-speed20000_eps.pth",
    r"C:\Users\admin\Desktop\IDP\CDP-CFD\arcs\code\observers\SO2\2024-10-13-16-08-18-SO2-Model-sn_scale-None-yummy-moss20000_eps.pth",
    r"C:\Users\admin\Desktop\IDP\CDP-CFD\arcs\code\observers\neuralswarm\2024-10-20-11-29-07-NSwarm-Model-sn_scale-4-144k-datapoints-corrected-bias-timid-pocket20000_eps.pth", 
]

models = []

model = DWPredictor()
model.load_state_dict(torch.load(model_paths[0], weights_only=True))
models.append(model)

model = ShallowEquivariantPredictor()
model.load_state_dict(torch.load(model_paths[1], weights_only=True))
models.append(model)

model = NeuralSwarmPredictor()
model.load_state_dict(torch.load(model_paths[2], weights_only=True))
models.append(model)

predictions = evaluate_zy_force_curvature(models, np.array(rel_state_vector_list))
labels = ["NDP with SN<4", "SO2-Equiv.", "Nrl.Swarm 2 UAV"]



for idx, prediction in enumerate(predictions):
    print("plotting pred.")
    plt.plot(time_seq, prediction[:,2], label=labels[idx])


plt.xlabel("time (s)")
plt.ylabel("Force (N)")
plt.title(f"Measured and predicted DW forces at\nSpeed of Y={y_velocity} (Both Hovering)")






plt.legend()
plt.show()


