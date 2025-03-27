from simcontrol import simcontrol2
import numpy as np
nan = float('NaN')


class UAVSimulator:
    def __init__(self):
        # init simcontrol
        self.port = 25556
        self.controller = simcontrol2.Controller("localhost", self.port)
        self.controller.clear()
        self.controller.start()
        self.state = None  # Current state of the UAV

        # init time params
        self.sim_max_duration = 1.5 # in seconds
        self.control_frequency = 200 # in Hz
        self.time_step = self.controller.get_time_step()
        self.total_sim_time_steps = self.sim_max_duration / self.time_step
        self.steps_per_call = int(1.0 / self.control_frequency / self.time_step)
        
        # init sensors
        self.imu1_idx = self.controller.get_sensor_info('imu1').index
        self.imu2_idx = self.controller.get_sensor_info('imu2').index
        # init actuators
        # px4 controller handle
        self.px4_input1_idx = self.controller.get_actuator_info('controller1').index
        # RL agent's UAV2 rotor actuators handles
        self.uav_2_r1_idx = self.controller.get_actuator_info("uav_2_r1_joint_motor").index
        self.uav_2_r2_idx = self.controller.get_actuator_info("uav_2_r2_joint_motor").index
        self.uav_2_r3_idx = self.controller.get_actuator_info("uav_2_r3_joint_motor").index
        self.uav_2_r4_idx = self.controller.get_actuator_info("uav_2_r4_joint_motor").index

        self.reset()       # Initialize the simulation
        
    def reset(self):
        """Reset the simulator to the initial state"""
        self.controller.clear()
        self.controller.start()
        self.curr_sim_time = 0.0
        self.curr_step = 0
        self.done = False
        return np.array([0., 0., 0., 0.,-1.5,0., 0.6, 0., 0.], dtype=np.float32) # TODO init first state with positions of uavs, vel., etc.

    
    def step(self, action):
        """Take a step in the simulation with the given action"""
        # Apply the action to the simulator
        self.take_action(action)
        
        # Retrieve the new state and check if the episode is done
        self.state = self.get_state()
        
        # advance timer and step counter:
        self.curr_sim_time += self.steps_per_call * self.time_step
        self.curr_step += self.steps_per_call

        reward = self.calculate_reward(self.state)  
        self.done = self.check_done()  # Check if episode has ended
        truncated = False # no logic here yet
        info = self.curr_sim_time

        return self.state, reward, self.done, info
    
    def take_action(self, action):
        """Apply the action to the UAV"""
        # set uav 1 to position_controller=0, x=0, y=0, z=0.5
        self.uav_1_destination = (0.0, 0.0, 0.0, 0.5, nan, nan, nan, nan, nan, nan, 0.0, nan)
        self.reply = self.controller.simulate(self.steps_per_call,  
        { 
            self.px4_input1_idx: self.uav_1_destination,
            self.uav_2_r1_idx: (action[0],), 
            self.uav_2_r2_idx: (action[0],), 
            self.uav_2_r3_idx: (action[0],),  # front right (neg for upwards)
            self.uav_2_r4_idx: (action[0],),  # back left (neg for upwards)
        })
        
    def get_state(self):
        """Return the nessarary states of the UAVs and time"""
        imu_other = self.reply.get_sensor_output(self.imu1_idx)
        imu_self  = self.reply.get_sensor_output(self.imu2_idx)

        # return steering UAVs x,z pos. and y vel.
        xz_pos_y_vel_self = np.array([imu_self[0], imu_self[2], imu_self[7]])

        # return as well rel. pos and vel. of neighbor UAV
        rel_pos_vec = np.array(imu_other[:3]) - np.array(imu_self[:3])
        rel_vel_vec = np.array(imu_other[6:9]) - np.array(imu_self[6:9])


        # return observation as well time and steps
        observation = np.concatenate((xz_pos_y_vel_self, rel_pos_vec, rel_vel_vec), dtype=np.float32)
        print("#### obser", observation)
        return observation
    
    def calculate_reward(self, state):
        """Calculate reward based on the state"""
        reward = 0
        vy_ref = 2.0

        x_err = -(state[0]**2)
        z_err = -(state[1]**2)
        vy_err = -np.square(vy_ref - state[2])
        collision_penalty = -100 if np.linalg.norm(state[3:6]) < 0.1 else 0 # if collision with other uav

        reward  = x_err + z_err + vy_err + collision_penalty
        return reward
    
    def check_done(self):
        """Check if the simulation episode is done"""

        if not self.curr_sim_time < self.sim_max_duration: 
            # time exceeded, close ar controller and return
            #self.controller.clear()
            
            return True
        else: 
            return False

    def close(self):
        self.controller.clear()
        self.controller.close()
    
    

