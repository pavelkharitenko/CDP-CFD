from simcontrol import simcontrol2
nan = float('NaN')


class UAVSimulator:
    def __init__(self):
        self.port = 25556
        self.control_frequency = 200 # in Hz
        self.sim_max_duration = 0.8 # in seconds
        self.state = None  # Current state of the UAV
        self.done = False  # Simulation end flag
        self.reset()       # Initialize the simulation
        
    def reset(self):
        """Reset the simulator to the initial state"""
        self.state = self.get_initial_state()
        self.done = False
        return self.state

    def get_initial_state(self):
        """Init interface to AR, to AR's UAV's actuators, sensors, etc."""
        # get input/output control handle from AR sim
        self.controller = simcontrol2.Controller("localhost", self.port)
        self.controller.start()

        # init time params
        self.time_step = self.controller.get_time_step()
        self.total_sim_time_steps = self.sim_max_duration / self.time_step
        self.steps_per_call = int(1.0 / self.control_frequency / self.time_step)
        self.curr_sim_time = 0.0
        self.curr_step = 0
        

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

        return None # TODO init first state with positions of uavs, vel., etc.

    
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
        info = {}
        return self.state, reward, self.done, truncated, info
    
    def take_action(self, action):
        """Apply the action to the UAV"""
        # set uav 1 to position_controller=0, x=0, y=0, z=0.5
        self.uav_1_destination = (0.0, 0.0, 0.0, 0.5, nan, nan, nan, nan, nan, nan, 0.0, nan)
        self.reply = self.controller.simulate(self.steps_per_call,  
        { 
            self.px4_input1_idx: self.uav_1_destination,
            self.uav_2_r1_idx: (400,), 
            self.uav_2_r2_idx: (400,), 
            self.uav_2_r3_idx: (-400,),  # front right (neg for upwards)
            self.uav_2_r4_idx: (-400,),  # back left (neg for upwards)
        })
        
    def get_state(self):
        """Return the current state of the UAV"""
        imu1_data  = self.reply.get_sensor_output(self.imu1_idx)
        imu2_data  = self.reply.get_sensor_output(self.imu2_idx)


        
        return [imu1_data, imu2_data], self.curr_sim_time, self.curr_step
    
    def calculate_reward(self, state):
        """Calculate reward based on the state"""
        # Define reward function here based on the state
        reward = 0
        # For example: reward = -1 if UAV crashes, reward = 10 if UAV reaches goal, etc.
        return reward
    
    def check_done(self):
        """Check if the simulation episode is done"""

        if not self.curr_sim_time < self.sim_max_duration: 
            # time exceeded, close ar controller and return
            self.controller.clear()
            self.controller.close()
            return True
        else: 
            return False
    
    

