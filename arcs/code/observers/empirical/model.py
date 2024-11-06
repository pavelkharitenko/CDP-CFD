#----------------------------------
# Empirical DW model implemented from Jain et al. 
# Modeling of aerodynamic disturbances for proximity flight of multirotors
#----------------------------------
import numpy as np

class EmpiricalPredictor():
    def __init__(self):
        # params for dw model due to drag
        self.Aq = 0.0552503 # uav surface area, square meters
        self.Cd = 1.18 # aerodynamic property of uav (like flat plane)
        self.Ap = 1 # propeller disk area
        self.rho = 1.0 # density of air

        self.l = 0.3 # uav arm length
        self.L = 2.0*self.l # vehicle size
        self.c_ax = 1.0 # axial constant
        self.z0 = 3.0*self.L # virtual origin

        self.c_rad = 1.0
        
        

    def velocity_center(self, z):
        induced_vel = np.sqrt(self.T / (2*self.rho*self.Ap))
        Vmax = induced_vel* self.c_ax*self.L / (z - self.z0)

        return Vmax

    def velocity_field(self, z,r):
        vel = self.velocity_center(z)*np.exp(-self.c_rad*np.square(r/z - self.z0))
        
        return vel
    
    def F_drag(self,z,r, grid_points, cell_size, uav_1, uav_2):
        f_int = 0
        A_cell = cell_size**2
        #grid_points = [[-0.088528,  0.1035  ], ...]
                        
        
        for grid_pnt in grid_points:
            # TODO compute absolute x,y gridpoint position, then substract to 
            x, y = uav_2.rot @ grid_pnt
            x_abs, y_abs = x + uav_2.pos.x, y + uav_2.pos.y
            # horizontal separation from uav 1
            r_dash = np.sqrt((uav_1.pos.x - x_abs)**2 + (uav_1.pos.y - y_abs) **2)
            # vertical separation from uav 1
            z_dash = uav_1.pos.z - uav_2.pos.z
            f_cell = A_cell * 0.5 * self.Cd * self.pho * self.velocity_field(z_dash, r_dash)**2
            f_int += f_cell
        

        return np.array([0.0,0.0,-1.0]) * f_int
    
    def T_drag(self, z, r, grid_points, cell_size, uav_1, uav_2):
        torque_int = 0
        A_cell = cell_size**2
        #grid_points = [[-0.088528,  0.1035  ], ...]
                        
        
        for grid_pnt in grid_points:
            # rotate gridpoint vector to uav frame
            x, y, z = uav_2.rot @ np.array([grid_pnt[0], grid_pnt[1], 0]) # rotate correctly to 
            x_abs, y_abs, z_abs = x + uav_2.pos.x, y + uav_2.pos.y, z + uav_2.pos.z
            z_dash = uav_1.pos.z - uav_2.pos.z
            # horizontal separation from uav 1
            r_dash = np.sqrt((uav_1.pos.x - x_abs)**2 + (uav_1.pos.y - y_abs) **2)
            
            # vertical separation from uav 1
            z_dash = uav_1.pos.z - uav_2.pos.z

            # distance uav 2 center and gridpoint
            r_dash_abs = np.array([x_abs, y_abs, z_abs])
            r = np.array([uav_2.pos.x, uav_2.pos.y, uav_2.pos.z])
            
            # torque arm
            t_arm = r_dash_abs - r

            # vertical separation
            torque_cell = t_arm @ [0,0,-1]*A_cell * 0.5 * self.Cd * self.pho * self.velocity_field(z_dash, r_dash)**2
            torque_int += torque_cell
        
        return torque_int

    def propeller_thrust_change(self, prop_speed, oncoming_vel):
        
        #bv is thrust_delay_coefficient
        bv = 3.7 * 10e-7
        return -bv * prop_speed * oncoming_vel**2
    
    def propeller_thrust_change_torque(self):
        pass


    def __call__(self, uav_1_x, uav_2_x):
        return np.array(uav_2_x) - np.array(uav_1_x)


    def evaluate(self, rel_state_vectors):
        """
        From R:6 (rel_pos, rel_vel) to R:3 (0, 0, dw_z) 
        """
        with torch.no_grad():
            inputs = torch.tensor(rel_state_vectors).to(torch.float32)
            dw_forces = self.forward(inputs).detach().cpu().numpy()
            
            padding = np.zeros((len(dw_forces), 3))
            padding[:,2] = dw_forces.squeeze()
            return padding


ep = EmpiricalPredictor()

print(ep([0,0,0], [0,9,9]))