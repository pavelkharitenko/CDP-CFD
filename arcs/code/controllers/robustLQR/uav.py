import numpy as np

class UAV:
    def __init__(self, name, controller, controller_name, imu_name, jft_sensor_names, dt=0.01):


        self.name = name
        self.controller = controller
        self.controller_name = controller_name
        self.imu_name = imu_name
        self.jft_sensor_names = jft_sensor_names
        self.dt = dt
        
        self.x = np.zeros(7) # State vector [p_x, p_y, p_z, v_x, v_y v_z, yaw]
        self.states = []
        self.accelerations = []
        self.orientations = []
        self.jft_forces_list = []
        self.timestamp_list = []

        # State-space matrices (from the image)
        self.A = np.block([
            [np.zeros((3, 3)), np.eye(3), np.zeros((3, 1))],
            [np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 1))],
            [np.zeros((1, 3)), np.zeros((1, 3)), np.zeros((1, 1))]
        ])

        self.B = np.block([
            [np.zeros((3, 3)), np.zeros((3, 1))],
            [np.eye(3), np.zeros((3, 1))],
            [np.zeros((1, 3)), np.ones((1, 1))]
        ])

        self.C = np.eye(7)  # Full state observation

        self.init_sensors()

    def state_update(self, u):
        """
        Update the state based on the control input using the state-space model.
        
        Args:
            u (np.array): Control input [a_n, a_e, a_d, dot_psi].
        
        Returns:
            np.array: Updated state vector.
        """
        # Update the state using the discrete-time approximation
        dx = self.A @ self.x + self.B @ u
        self.x += dx * self.dt
        return self.x

    def output(self):
        """
        Get the observable state (y = Cx).
        
        Returns:
            np.array: Observable state vector.
        """
        return self.C @ self.x

    def reset(self, initial_state=None):
        """
        Reset the UAV to an initial state.
        
        Args:
            initial_state (np.array): Initial state to reset to.
        """
        if initial_state is not None:
            self.x = initial_state
        else:
            self.x = np.zeros(7)

    # AEROCAE related
    
    def init_sensors(self):
        self.px4_idx = self.controller.get_actuator_info(self.controller_name).index
        self.imu_idx = self.get_sensor_idx(self.imu_name)
        self.jtf_sensor_idxs = self.get_sensor_idx(self.jft_sensor_names)
    
    def update(self, reply, timestep):
        self.reply = reply
        # read pos,vel,acc vectors
        self.x = self.read_sensor(reply, self.imu_idx, [0,1,2, 6,7,8, 3])
        self.states.append(self.x)
        self.orientations.append(self.read_sensor(reply, self.imu_idx, [3,4,5,18,19,20,21]))
        self.accelerations.append(self.read_sensor(reply, self.imu_idx, [12,13,14]))


        # update jft sensors of body and 5 rotors
        r1_r2_r3_r4_jft = self.read_multiple_sensors(reply, self.jtf_sensor_idxs, [2])
        self.jft_forces_list.append(r1_r2_r3_r4_jft)

        # update current time
        self.timestamp_list.append(timestep)

    def get_sensor_idx(self, sensor_names):
        """
        Returns sensor index or list of sensor indices if list of names is passed.
        """
        if isinstance(sensor_names, list):
            result = []
            for sensor_name in sensor_names:
                #print(sensor_name, self.controller.get_sensor_info(sensor_name).index)
                result.append(self.controller.get_sensor_info(sensor_name).index)
            return result
        else:
            return self.controller.get_sensor_info(sensor_names).index



    def read_sensor(self, reply, sensor_idx, indicies=[0]):
        """
        Read current value tuple of a sensor or at specified indices of that tuple
        """
        result = []
        sensor_tuple_data = reply.get_sensor_output(sensor_idx)

        for i in indicies:
            #print("### sensor tuple of ", sensor_idx, sensor_tuple_data)
            result.append(sensor_tuple_data[i])
        if len(result) == 1:
            return result[0]
        else:
            return result



    def read_multiple_sensors(self, reply, sensor_indices, tuple_indices=None):
        """
        Read multiple sensor tuples, or each at tuple_indices of each tuple
        """
        result = []
        if not tuple_indices:
            for sensor_idx in sensor_indices:
                result.append(self.read_sensor(reply, sensor_idx))
        else:
            for sensor_idx in sensor_indices:
                result.append(self.read_sensor(reply, sensor_idx, tuple_indices))
        return result

