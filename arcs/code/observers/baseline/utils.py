import torch, pickle, randomname
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

def plot_xy_slices(model):
    """
    Visualize (x,y,z) output of NN as 2D heatmap. Based on "plot_historgrams" at
    https://github.com/Li-Jinjie/ndp_nmpc_qd/blob/master/ndp_nmpc/scripts/dnwash_nn_est/nn_train.py
    """
    
    # Plot xy-slices at different Z-values:
    xy_plane_z_samples = [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5]
    xy_range = 1.0 # size of XY plane
    plane_res = 200 # number of points sampled for plotting in one dimension

    color="autumn_r"
    test_f = []
    fig, ax = plt.subplots(2, len(xy_plane_z_samples), sharex=True, sharey=True, figsize=(6, 6))

    model.eval()
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    
    plotted_forces = [] # save plotted forces for adjusting min and max value of heatmaps

    all_imgs = [] # save all

    # loop over Z-heights and generate plane_res*plane_res sample points
    for idx, z_point in enumerate(xy_plane_z_samples):
        
        xy_samples = np.linspace(start=-xy_range, stop=xy_range, num=plane_res)
        sample_matrix = np.zeros([plane_res**2, 6])

        for i in range(plane_res):
            for j in range(plane_res):
                sample_matrix[i * plane_res + j, 0] = xy_samples[i]
                sample_matrix[i * plane_res + j, 1] = xy_samples[j]
                sample_matrix[i * plane_res + j, 2] = z_point
        
        
        sample_tensor = torch.from_numpy(sample_matrix).to(torch.float32)
        input = torch.autograd.Variable(sample_tensor).cuda()
        output = model(input)
        test_f.append(output)
        zs = output[:, 2]
        
        # add all encountered z-forces and save them for evaluating 
        plotted_forces.extend(zs.detach().cpu().numpy())


        plot_f = np.zeros((plane_res, plane_res))

        for i in range(plane_res):
            for j in range(plane_res):
                plot_f[i, j] = zs[i * plane_res + j]

        im = ax[0][idx].imshow(plot_f, extent=[-xy_range, xy_range, xy_range, -xy_range], cmap=color)
        all_imgs.append(im)
        ax[0][idx].set_title(f"Z = {z_point}m")
        
    
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)


    # plot xz-slice

    for k, test_y in enumerate(xy_plane_z_samples):
        
        test_xz = np.linspace(start=-xy_range, stop=xy_range, num=plane_res)
        test_matrix = np.zeros([plane_res**2, 6])
        # matrix actually (y,x,z) of state vector:
        for i in range(plane_res):
            for j in range(plane_res):
                test_matrix[i * plane_res + j, 0] = test_y
                test_matrix[i * plane_res + j, 1] = test_xz[i]
                test_matrix[i * plane_res + j, 2] = test_xz[j]
        
        
        torch_matrix = torch.from_numpy(test_matrix).to(torch.float32)
        input = torch.autograd.Variable(torch_matrix).cuda()
        output = model(input)
        test_f.append(output)
        ys = output[:, 1]
        
        plotted_forces.extend(ys.detach().cpu().numpy())

        plot_f = np.zeros((plane_res, plane_res))
        for i in range(plane_res):
            for j in range(plane_res):
                plot_f[i, j] = ys[i * plane_res + j]

        im = ax[1][k].imshow(plot_f, extent=[-xy_range, xy_range, xy_range, -xy_range], cmap="autumn_r")
        all_imgs.append(im)
        ax[1][k].set_title(f"Y = {test_y}m")

    # add the bar and fix min-max colormap
    for im in all_imgs:
        im.set_clim(vmin=np.min(plotted_forces), vmax=np.max(plotted_forces))

    fig.suptitle('\n Predicted Downwash Forces acting on Sufferer UAV \n Other drone is Z and Y meters above and right of Sufferer \n (trained on dummy data)')
    plt.xlabel('Forces in Newton (N)')
    #plt.ylabel('dummy data')
    fig.savefig('DW_Predictor_z_y_slices.png')
    plt.show()



def plot_3D_forces(model):

    """
    Visualize (x,y,z) output of NN as 2D heatmap. Based on "plot_historgrams" at
    https://github.com/Li-Jinjie/ndp_nmpc_qd/blob/master/ndp_nmpc/scripts/dnwash_nn_est/nn_train.py
    """
    
    # Plot xy-slices at different Z-values:
    xy_range = 1.0 # size of XY plane
    plane_res = 6 # number of points sampled for plotting in one dimension

    color="autumn_r"
    test_f = []
    #fig, ax = plt.subplots(2, len(xy_plane_z_samples), sharex=True, sharey=True, figsize=(6, 6))

    model.eval()
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    
    plotted_forces = [] 
    all_imgs = []

    # create i x j x k matrix to sample from network forces
    xy_samples = np.linspace(start=-xy_range, stop=xy_range, num=plane_res)
    sample_matrix = np.zeros([plane_res**3, 6])

    for i in range(plane_res):
        for j in range(plane_res):
            for k in range(plane_res):
                sample_matrix[i * plane_res + j + k* plane_res**2, 0] = xy_samples[i]
                sample_matrix[i * plane_res + j + k* plane_res**2, 1] = xy_samples[j]
                sample_matrix[i * plane_res + j + k* plane_res**2, 2] = xy_samples[k]
    
    
    sample_tensor = torch.from_numpy(sample_matrix).to(torch.float32)
    input = torch.autograd.Variable(sample_tensor).cuda()
    xyz_forces = model(input)
    
    # plot result
    input = input.to("cpu").detach().numpy()
    xyz_forces = xyz_forces.to("cpu").detach().numpy()

    plot_3d_vectorfield(input, xyz_forces, 1.0, "\n Predicted Downwash Force Vector Field \n (Force length adjusted)")
    

def plot_3d_vectorfield(startpositions, vectors, scale, title):
    # normalize for plotting
    #vectors = vectors / np.linalg.norm(vectors)
    # iterate over input and output and create 3D plot

    
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlabel('X') #, linespacing=4)
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    xs = [xyz[0] for xyz in startpositions]
    ys = [xyz[1] for xyz in startpositions]
    zs = [xyz[2] for xyz in startpositions]
    xmin, xmax = np.min(xs), np.max(xs)
    ymin, ymax = np.min(ys), np.max(ys)
    zmin, zmax = np.min(zs), np.max(zs)

    ax.set_xlim([zmin*1.2,zmax*1.2])
    ax.set_ylim([zmin*1.2,zmax*1.2])
    ax.set_zlim([zmin*1.2,zmax*1.2])

    for i in range(len(startpositions)):
        x,y,z = startpositions[i][:3]
        fx,fy,fz =  vectors[i] * scale # vector length factor for visualization
        ax.quiver(x, y, z, fx, fy, fz, color='steelblue')

    fig.suptitle(title)
    plt.show()
    #plt.savefig("3D_DW_dummydata.png")


def plot_NN_training(train_errors, val_errors):
    t = range(len(train_errors))
    plt.plot(t,train_errors)
    plt.plot(t,val_errors)
    plt.xlabel("Epoch i")
    plt.ylabel("Errors")
    plt.legend()
    plt.show()


def plot_static_dw_collection(state_sufferer, state_producer, ef_sufferer, jft_sufferer, timestamp_list):
    fig, axes = plt.subplots(2, 2)
    color1 = 'tab:red'
    color2 = 'tab:brown'
    color3 = 'tab:cyan'
    color4 = 'tab:pink'
    color5 = 'tab:olive'
    color6 = 'tab:purple'

    # Plot Both drones position
    ax1 = axes[0][0]
    ax1.set_title('DW Producer & Sufferer Positions in the YZ-Plane')
    ax1.scatter([xyz[1] for xyz in state_producer], [xyz[2] for xyz in state_producer], label='Producer UAV')
    ax1.scatter([xyz[1] for xyz in state_sufferer], [xyz[2] for xyz in state_sufferer], label='Sufferer UAV')
    ax1.set_xlabel('Y-Position (m)')
    ax1.set_ylabel('Z-Position (m)')
    ax1.legend()
    
    # Plot recorded forces over time
    ax2 = axes[1][0]
    ax2.set_title('External Forces on Sufferer UAV, EF and JFT sensors')
    ax2.set_xlabel('Y-Position (m) of Producer')
    ax2.set_ylabel('Force (N)')
    ax2.plot([xyz[1] for xyz in state_producer], [xyz[0] for xyz in jft_sufferer], label='x force (jft)')
    ax2.plot([xyz[1] for xyz in state_producer], [xyz[1] for xyz in jft_sufferer], label='y force (jft)')
    ax2.plot([xyz[1] for xyz in state_producer], [xyz[2] for xyz in jft_sufferer], label='z force (jft)')
    ax2.plot([xyz[1] for xyz in state_sufferer], ef_sufferer, label='z force (ef)')
    ax2.legend()

    ax3 = axes[0][1]
    ax3.set_title('DW Producer & Sufferer Positions in the XY-Plane')
    ax3.scatter([xyz[1] for xyz in state_producer], [xyz[0] for xyz in state_producer], label='Producer UAV')
    ax3.scatter([xyz[1] for xyz in state_sufferer], [xyz[0] for xyz in state_sufferer], label='Sufferer UAV')
    ax3.set_xlabel('Y-Position (m)')
    ax3.set_ylabel('X-Position (m)')
    ax3.legend()

    
    ax4 = axes[1][1]
    ax4.set_title('ExtForce vs JointForTor sensor Z force comparison')
    ax4.set_xlabel('time (s)')
    ax4.set_ylabel('Force (N)')
    ax4.plot(timestamp_list, ef_sufferer, label='ExtForSen', color=color4)
    ax4.plot(timestamp_list, [xyz[2] for xyz in jft_sufferer], label='JointForTor', color=color5)
    ax4.legend()


def init_experiment(env_name):  
    experiment_name = datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "-" + env_name + "-"+ randomname.get_name()
    print("###############################################")
    print("#### -- Beginning Experiment '", experiment_name, "' -- ####")
    return experiment_name


def save_experiment(exp_name, uav_list, success, sim_duration):
    exp_name += "-" + str(sim_duration) + "sec-" + str(len(uav_list[0].timestamp_list)) + "-ts"
    
    if not success:
        print("#### --- during simulation, an error occured, not saving results of experiment", exp_name)
        exit(1)
    else:
        print("#### Saving experiment", exp_name, "...")
        
        exp_obj = {'exp_name': exp_name, 'uav_list': uav_list}

        save_name = exp_name + '.p'
        with open(save_name, 'wb') as handle:
            pickle.dump(exp_obj, handle)
        

        print("#### Saved.")
        print("################################")
        return save_name



        


def load_forces_from_dataset(exp_pkl_path):
    with open(exp_pkl_path, 'rb') as handle:
            experiment = pickle.load(handle)
    print("Loaded " + experiment["exp_name"])
    return experiment

def extract_labeled_dataset_ndp(uav_list):
    uav_1, uav_2 = uav_list
    rel_state_list = [rel_state[0:6] for rel_state in np.array(uav_1.states) - np.array(uav_2.states)]
    #print(uav_2.jft_forces_list)

    # Compute from F_uav = -mg + Fu + bias + Fd     => Fd = F_uav + mg - Fu - bias
    Fz_total = [uav_2.total_mass * state[8] for state in uav_2.states] # recorded actual uav m*a z-Force
    Fg = [uav_2.total_mass * 9.81 for state in uav_2.states] # uav gravitational force m*g
    Fz_u_total = np.array([-np.sum(body_r1_r2_r3_r4, axis=0) for body_r1_r2_r3_r4 in uav_2.jft_forces_list])
    bias = 4.5 # on average, Fu compentates always more by 5N, so it is not disturbance force
    
    print(len(Fz_total))
    print(len(Fz_u_total))
    print(len(Fg))

    dw_forces = np.array(Fz_total) + np.array(Fg) - np.array(Fz_u_total)
    dw_forces -= bias

    dw_force_vectors = np.array([(0,0,dw_force) for dw_force in dw_forces])

    print(dw_force_vectors)

    xy_data = list(zip(rel_state_list, dw_force_vectors))
    print("data extracted:", xy_data[-6:])

    
    
    plot_3d_vectorfield(rel_state_list, dw_force_vectors,1.0/np.max(np.abs(dw_force_vectors)), "Created z-force dataset from collected data")


    


