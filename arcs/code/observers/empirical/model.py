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
    
    def F_drag(self,z,r, grid_points, cell_size):
        f_int = 0
        A_cell = cell_size**2
        grid_points = [[-0.088528,  0.1035  ],
                        [-0.118528,  0.1335  ],
                        [-0.148528,  0.1635  ],
                        [-0.208528,  0.1935  ],
                        [-0.178528,  0.1935  ],
                        [-0.208528,  0.2235  ]]
        
        for grid_pnt in grid_points:
            # TODO compute absolute x,y gridpoint position, then substract to 
            x, y = grid_pnt
            radius = np.sqrt(x**2 + y**2)
            f_cell = A_cell * 0.5 * self.Cd * self.pho * self.velocity_field(relative_z, radius)
            f_int += f_cell
        

        return np.array([0.0,0.0,-1.0]) * f_int


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