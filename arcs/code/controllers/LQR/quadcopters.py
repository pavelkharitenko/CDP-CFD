import numpy as np

class UAV_:
    def __init__(self, mass=1.0, inertia=None, initial_state=None):
        """
        Initialize the UAV quadrotor model.
        :param mass: Mass of the quadrotor (kg)
        :param inertia: Inertia matrix of the quadrotor (3x3 numpy array)
        :param initial_state: Initial state [x, y, z, vx, vy, vz, roll, pitch, yaw, p, q, r]
        """
        self.mass = mass
        self.inertia = inertia if inertia is not None else np.diag([0.1, 0.1, 0.1])  # Example inertia values
        self.state = initial_state if initial_state is not None else np.zeros(12)  # [x, y, z, vx, vy, vz, roll, pitch, yaw, p, q, r]
        
        # PX4 Control inputs
        self.throttle = 0  # Throttle input (0 to 1)
        self.roll_torque = 0  # Relative roll torque
        self.pitch_torque = 0  # Relative pitch torque
        self.yaw_torque = 0  # Relative yaw torque

        self.A = self.compute_A()  # State matrix
        self.B = self.compute_B()  # Control matrix

    def compute_A(self):
        """
        Compute the state matrix A based on UAV dynamics.
        Linearized around hover.
        """
        g = 9.81  # Acceleration due to gravity
        return np.array([
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, g, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, -g, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        ])

    def compute_B(self):
        """
        Compute the control matrix B based on UAV dynamics.
        """
        I_x, I_y, I_z = self.inertia
        return np.array([
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [1/self.mass, 0, 0, 0],
            [0, 1/I_x, 0, 0],
            [0, 0, 1/I_y, 0],
            [0, 0, 0, 1/I_z]
        ])


    def update_control_inputs(self, throttle, roll_torque, pitch_torque, yaw_torque):
        """
        Update the control inputs for the UAV.
        :param throttle: Throttle input (0 to 1), mapped to thrust
        :param roll_torque: Relative roll torque input
        :param pitch_torque: Relative pitch torque input
        :param yaw_torque: Relative yaw torque input
        """
        self.throttle = throttle
        self.roll_torque = roll_torque
        self.pitch_torque = pitch_torque
        self.yaw_torque = yaw_torque

    def dynamics(self, dt=0.01):
            """
            Update the UAV state based on the current throttle and relative torques.
            """
            # Constants
            gravity = np.array([0, 0, -9.81])  # Gravity vector (m/s^2)
            
            # Thrust force in body frame
            thrust_force_body = np.array([0, 0, self.throttle * self.mass * 9.81])
            
            # Rotation matrix based on current roll, pitch, yaw
            roll, pitch, yaw = self.state[6], self.state[7], self.state[8]
            R = self.rotation_matrix(roll, pitch, yaw)
            
            # Thrust force in world frame
            thrust_force_world = R.dot(thrust_force_body)
            
            # Net force (thrust + gravity)
            net_force = thrust_force_world + self.mass * gravity
            
            # Linear acceleration (world frame)
            acceleration = net_force / self.mass
            
            # Update velocity and position
            velocity = self.state[3:6]  # [vx, vy, vz]
            new_velocity = velocity + acceleration * dt
            new_position = self.state[0:3] + new_velocity * dt
            
            # Calculate angular acceleration from relative torques
            angular_velocity = self.state[9:12]  # [p, q, r]
            torques = np.array([self.roll_torque, self.pitch_torque, self.yaw_torque])
            angular_acceleration = np.linalg.inv(self.inertia).dot(torques - np.cross(angular_velocity, self.inertia.dot(angular_velocity)))
            
            # Update angular velocity and orientation
            new_angular_velocity = angular_velocity + angular_acceleration * dt
            new_orientation = self.state[6:9] + new_angular_velocity * dt
            
            # Update the state vector
            self.state = np.concatenate((new_position, new_velocity, new_orientation, new_angular_velocity))


    def update_state(self, control_input, dt):
        """
        Apply the control input to the UAV, then update the dynamics.
        control_input: [roll_torque, pitch_torque, yaw_torque, throttle]
        dt: Time step for the dynamics update.
        """
        # Apply control inputs
        self.roll_torque = control_input[0]
        self.pitch_torque = control_input[1]
        self.yaw_torque = control_input[2]
        self.throttle = control_input[3]

        # Update the dynamics with the new control inputs
        self.dynamics(dt)



    def update_state_from_imu(self, imu_state):
        """
        Update the UAV state based on IMU readings.
        imu_state: list or array containing [x, y, z, vx, vy, vz, roll, pitch, yaw, p, q, r]
        """
        self.state = imu_state

    def rotation_matrix(self, roll, pitch, yaw):
        """
        Return the rotation matrix based on the current roll, pitch, and yaw angles.
        """
        c_roll = np.cos(roll)
        s_roll = np.sin(roll)
        c_pitch = np.cos(pitch)
        s_pitch = np.sin(pitch)
        c_yaw = np.cos(yaw)
        s_yaw = np.sin(yaw)

        R = np.array([
            [c_yaw * c_pitch, c_yaw * s_pitch * s_roll - s_yaw * c_roll, c_yaw * s_pitch * c_roll + s_yaw * s_roll],
            [s_yaw * c_pitch, s_yaw * s_pitch * s_roll + c_yaw * c_roll, s_yaw * s_pitch * c_roll - c_yaw * s_roll],
            [-s_pitch, c_pitch * s_roll, c_pitch * c_roll]
        ])
        
        return R


    def reset(self, initial_state=None):
        """
        Reset the UAV state to initial conditions.
        :param initial_state: New state [x, y, z, vx, vy, vz, roll, pitch, yaw, p, q, r]
        """
        self.state = initial_state if initial_state is not None else np.zeros(12)

    def get_state(self):
        """
        Get the current state of the UAV.
        :return: Current state as a numpy array
        """
        return self.state
