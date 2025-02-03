import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../../utils/')
from utils import *


from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tabulate import tabulate

sys.path.append('../../observers/')

from neuralswarm.model import NeuralSwarmPredictor
from ndp.model import DWPredictor
from SO2.model import ShallowEquivariantPredictor
from agile.model import AgileContinousPredictor, AgileShallowPredictor
from NNARX.model import NNARXModel
#from model import DWPredictor


roll_iterations = False

dataset_paths = [
        find_file_with_substring("raw_data_1_flybelow_200Hz_80_005_len34951ts_51_iterations_testset.npz"),
        find_file_with_substring("raw_data_2_flyabove_200Hz_80_005_len32956ts_46_iterations_testset.npz"),
        find_file_with_substring("raw_data_3_swapping_200Hz_80_005_len29736ts_52_iterations_testset.npz"),
        find_file_with_substring("raw_data_3_swapping_fast_200Hz_80_005_len20802ts_67_iterations_testset.npz"),
]

dataset_titles = [
    "1 fly below",
    "2 fly above",
    "3 swapping",
    "3 swapping (very fast)",
    "all 4 in total"
]





for title_idx, path_idx in enumerate([[0],[1],[2],[3],[0,1,2,3]]):

    uav_1_states = []
    uav_2_states = []
    time_seq = []



    for path in np.array(dataset_paths)[path_idx]:
        
        data = np.load(path)
        uav_1_state, uav_2_state = data['uav_1_states'][:-1], data['uav_2_states'][:-1]

        uav_1_states.extend(uav_1_state)
        uav_2_states.extend(uav_2_state)
        time_seq.extend(np.arange(0,len(uav_1_state)))

    print("dataset length", len(uav_1_states))
    rel_state_vector_list = np.array(uav_1_states) - np.array(uav_2_states)
    _, overlap_indices, overlaps = extract_data(uav_1_states, uav_2_states, time_seq)
    u2_x_dw_forces, u2_y_dw_forces, u2_z_dw_forces = extract_dw_forces(uav_2_states)




    model_paths = [

    # NDP
    find_file_with_substring("raw-remote30"),  
    # Agile Cont.
    #find_file_with_substring("broad-ramp10000"),
    #find_file_with_substring("future-country30000"),
    #find_file_with_substring("polynomial-sectional20"),
    #find_file_with_substring("upbeat-elk30"),
    find_file_with_substring("crunchy-muffler200"),




    ]




    models = []

    colors= ["green", "purple", "grey", "red", "cyan", "pink", "black"]


    #model = DWPredictor()
    #model.load_state_dict(torch.load(model_paths[0], weights_only=True))
    #models.append(model)

    # model = NeuralSwarmPredictor()
    # model.load_state_dict(torch.load(model_paths[1], weights_only=True))
    # models.append(model)

    model = DWPredictor()
    model.load_state_dict(torch.load(model_paths[0], weights_only=True))
    models.append(model)

    #model_al1 = AgileContinousPredictor()
    #model_al1.load_state_dict(torch.load(model_paths[1], weights_only=True))

    #model_al4 = AgileShallowPredictor()
    #model_al4.load_state_dict(torch.load(model_paths[1], weights_only=True))

    model_al5 = AgileShallowPredictor(output_dim=3)
    model_al5.load_state_dict(torch.load(model_paths[1], weights_only=True))



    predictions = []
    predictions.append(model.evaluate(rel_state_vector_list[:,:6]))
    predictions.append(model_al5.evaluate(np.array(uav_1_states), np.array(uav_2_states)))



    labels = [ 
    "NDP batched 20e",
    "Shallow 30e",
    "Agile 20k full data",
    "NNARX 15",
    "shllw 10",
    "shllw 30",
    ]


    all_results = []
    
   

    #fig.plot(u2_z_forces, label="UAV's z-axis forces") # unsmoothed
    #plot_array_with_segments(fig, time_seq, smoothed_u2_z_forces, color="blue", roll=roll_iterations, label="UAV total Z-forces")
    #plot_array_with_segments(fig, time_seq, u2_thrusts,  color="orange", roll=roll_iterations, label="controller z-forces")


    fig = plt.subplot()
    plot_array_with_segments(fig, np.array(time_seq)[overlaps], u2_z_dw_forces[overlaps], color="magenta", roll=roll_iterations, label="downwash disturbance forces", overlaps=overlaps)

    # evaluate predictors
    for idx, prediction in enumerate(predictions):
        #scenario_title = dataset_titles[idx]
        #print("Now:", labels[idx])

        
        plot_array_with_segments(fig, np.array(time_seq)[overlaps], prediction[:,2][overlaps], roll=roll_iterations, color=colors[idx], label=labels[idx])
        #plot_array_with_segments(fig, np.array(time_seq)[overlaps], prediction[:,1][overlaps], roll=roll_iterations, color=colors[idx], label=labels[idx])

        #compute_rmse(prediction[:,2][overlaps], u2_z_dw_forces[overlaps],label=labels[idx])
        #print(f"R2 score of {labels[idx]}", r2_score(u2_z_dw_forces[overlaps],prediction[:,2][overlaps]))

        # Compute metrics
        metrics = compute_metrics(u2_z_dw_forces[overlaps], prediction[:, 2][overlaps])
        #metrics = compute_metrics(u2_y_dw_forces[overlaps], prediction[:, 1][overlaps])

        metrics["Label"] = labels[idx]
        all_results.append(metrics)
    

    # Print metrics as a table
    print(f"\nComparison of Prediction ({dataset_titles[title_idx]}):")
    headers = ["Label", "RMSE", "NRMSE (Range)", "MAE", "R2 Score"]
    table = [[res["Label"], res["RMSE"], res["NRMSE (Mean)"], res["MAE"], res["R2 Score"]] 
                for res in all_results]
    print(tabulate(table, headers=headers, tablefmt="grid"))

 


    plt.ylabel("Force [N]")
    plt.xlabel("time [s]")
    plt.grid()
    plt.title("Actual bottom UAV Z-forces, controller's thrust, and residual downwash force")
    plt.legend()
    plt.show()



# 4 create label:
u2_z_dw_forces = np.array([[0,0,z_force] for z_force in u2_z_dw_forces])

# label format v1: [state_uav1, state_uav2, dw_forces]
# find indices where the time array decreases (reset points) for number of iterations
reset_indices = np.where(np.diff(time_seq) < 0)[0] + 1  # Add 1 to shift to the start of the next segment
reset_indices = np.append([0], reset_indices)  # Include the start of the array as the first segment boundary
reset_indices = np.append(reset_indices, len(time_seq))  # Include the end of the array as the last boundary
n_itrs = len(reset_indices)