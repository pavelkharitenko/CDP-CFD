from simcontrol import simcontrol2
import numpy as np
import matplotlib.pyplot as plt


class UAV():
    def __init__(self,uav_name, controller, controller_name, imu_name, external_force_sensors_names, joint_force_torque_sensor_names, mounted_jft_sensor=None):
        # init properties
        self.total_mass = 3.3035
        self.name = uav_name
        self.controller = controller
        self.controller_name = controller_name
        self.imu_name = imu_name
        self.jft_sensor_names = joint_force_torque_sensor_names
        self.ef_sensor_names = external_force_sensors_names
        self.states = [] # contains Px, Py, Pz, Vx,Vy,Vz, Ax,Ay,Az, Yaw,Pitch,Roll, qw,qx,qy,qz  --> should also contain input u or force or each rotor in future
        self.jft_forces_list = []
        self.ext_forces_list = []
        self.timestamp_list = []
        self.mounted_jft_sensor = mounted_jft_sensor
        self.mounted_jft_sensor_list = []
        # init sensors
        self.init_sensors()


    def init_sensors(self):
        self.px4_idx = self.controller.get_actuator_info(self.controller_name).index
        self.imu_idx = self.get_sensor_idx(self.imu_name)
        self.ef_sensor_idxs = self.get_sensor_idx(self.ef_sensor_names)
        self.jtf_sensor_idxs = self.get_sensor_idx(self.jft_sensor_names)
        if self.mounted_jft_sensor:
            self.mounted_jft_sensor_idx = self.get_sensor_idx(self.mounted_jft_sensor)
        
    def update(self, reply, timestep):
        self.reply = reply
        # read pos,vel,acc vectors
        self.states.append(self.read_sensor(reply, self.imu_idx, [0,1,2, 6,7,8, 12,13,14, 3,4,5, 18,19,20,21]))
        # update external force sensors of body and 5 rotors
        ext_forces_body_r1234 = self.read_multiple_sensors(reply, self.ef_sensor_idxs)
        self.ext_forces_list.append(ext_forces_body_r1234)

        # update jft sensors of body and 5 rotors
        body_r1_r2_r3_r4_jft = self.read_multiple_sensors(reply, self.jtf_sensor_idxs, [2])
        self.jft_forces_list.append(body_r1_r2_r3_r4_jft)

        # update mounted jft sensor if available
        if self.mounted_jft_sensor:
            self.mounted_jft_sensor_list.append(self.read_sensor(reply, self.mounted_jft_sensor_idx, [0,1,2,3,4,5]))
            #print("### ### stored mtd sensor:", self.mounted_jft_sensor_list[-1][2])
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






def plot_uav_statistics(uav_list, begin=None, end=None):
    
    

    fig, axes = plt.subplots(3,len(uav_list))

    for i, uav in enumerate(uav_list):
        
        # filter for time
        timestamp_list = np.array(uav.timestamp_list)
        ext_forces_list = np.array(uav.ext_forces_list)
        jft_forces_list = np.array(uav.jft_forces_list)
        states = np.array(uav.states)
        if uav.mounted_jft_sensor:
            mounted_jft_meas = np.array([-sensor_tuple[2] for sensor_tuple in uav.mounted_jft_sensor_list])

        if begin:
            filter_condition = (timestamp_list> begin) & (timestamp_list < end)
            timestamp_list = timestamp_list[filter_condition]
            ext_forces_list = ext_forces_list[filter_condition]
            jft_forces_list = jft_forces_list[filter_condition]
            states = states[filter_condition]

            if uav.mounted_jft_sensor:
                mounted_jft_meas = mounted_jft_meas[filter_condition]


        # plot ef forces
        total_z_ef = [np.sum(ef) for ef in ext_forces_list]

        ef_ax = axes[0][i]
        ef_ax.set_title("Z-axis External Forces of '" + uav.name +  "' UAV")
        ef_ax.set_xlabel('time (s)')
        ef_ax.set_ylabel('Force (N)')
        ef_ax.plot(timestamp_list, [ef[0] for ef in ext_forces_list], label='body')
        ef_ax.plot(timestamp_list, [ef[1] for ef in ext_forces_list], label='r1')
        ef_ax.plot(timestamp_list, [ef[2] for ef in ext_forces_list], label='r2')
        ef_ax.plot(timestamp_list, [ef[3] for ef in ext_forces_list], label='r3')
        ef_ax.plot(timestamp_list, [ef[4] for ef in ext_forces_list], label='r4')
        ef_ax.plot(timestamp_list, total_z_ef, label='total')
        ef_ax.legend()

        # plot jft forces

        summed_z_forces_rotors = np.array([-np.sum(body_r1_r2_r3_r4, axis=0) for body_r1_r2_r3_r4 in jft_forces_list])
        uav_actual_z_forces = [uav.total_mass * state[8] for state in states]
        residual_z_forces = np.array(uav_actual_z_forces) - summed_z_forces_rotors
        uav_gravitational_force = [uav.total_mass * 9.81 for state in states]
        jft_gravity_force_difference = summed_z_forces_rotors - uav_gravitational_force

        jft_ax = axes[1][i]
        jft_ax.set_title("Z-axis JFT-sensor Measurments of '" + uav.name +  "' UAV")
        jft_ax.set_xlabel('time (s)')
        jft_ax.set_ylabel('Force (N)')
        #jft_ax.plot(timestamp_list, [body_r1_r2_r3_r4[0] for body_r1_r2_r3_r4 in jft_forces_list], label='z-axis body')
        #jft_ax.plot(timestamp_list, [body_r1_r2_r3_r4[1] for body_r1_r2_r3_r4 in jft_forces_list], label='z-axis imu')
        jft_ax.plot(timestamp_list, [-body_r1_r2_r3_r4[2] for body_r1_r2_r3_r4 in jft_forces_list], label='z-axis r1')
        jft_ax.plot(timestamp_list, [-body_r1_r2_r3_r4[3] for body_r1_r2_r3_r4 in jft_forces_list], label='z-axis r2')
        jft_ax.plot(timestamp_list, [-body_r1_r2_r3_r4[4] for body_r1_r2_r3_r4 in jft_forces_list], label='z-axis r3')
        jft_ax.plot(timestamp_list, [-body_r1_r2_r3_r4[5] for body_r1_r2_r3_r4 in jft_forces_list], label='z-axis r4')
        
        if uav.mounted_jft_sensor:
            jft_ax.plot(timestamp_list, mounted_jft_meas, label='mntd. jft sensor')

        jft_ax.plot(timestamp_list, summed_z_forces_rotors, label='jft z in total')
        jft_ax.plot(timestamp_list, uav_actual_z_forces, label='real z-frc. (m*z-acc)')
        jft_ax.plot(timestamp_list, residual_z_forces, label='mass x gravity')
        jft_ax.plot(timestamp_list, residual_z_forces, label='res. z-force (ma-jft)')
        jft_ax.legend()

        # plot statistics
        statistics = []
        mean_ef = np.mean(total_z_ef)
        mean_thrust = np.mean(summed_z_forces_rotors)
        mean_real_thrust_difference = np.mean(residual_z_forces)
        mean_thrust_mg_difference = np.mean(jft_gravity_force_difference)
        stats = [mean_ef, mean_thrust, mean_real_thrust_difference, mean_thrust_mg_difference]
        column_names = ("Avg. total ext. forces", "Avg. thrust of all rotors", 
                        "Avg. diff. real z-frc. - jft", "Avg. diff. jft - M*g")
        
        if uav.mounted_jft_sensor:
            column_names = ("Avg. extfor", "Avg. thrust", 
                        "Avg. real z-frc. - jft", "Avg. jft - M*g",
                        "Avg. real z-frc - mtd. jft")
            mean_real_thrust_mounted_jft = np.mean(uav_actual_z_forces - mounted_jft_meas)
            stats.append(mean_real_thrust_mounted_jft)
            
        
        stats = [np.round(stat,2) for stat in stats]
        statistics.append(stats)
        table_ax = axes[2][i]
        table_ax.axis('tight')
        table_ax.axis('off')
        stat_table = table_ax.table(cellText=statistics,colLabels=column_names, loc='center')
        stat_table.auto_set_font_size(False)
        stat_table.set_fontsize(8)
        

        




