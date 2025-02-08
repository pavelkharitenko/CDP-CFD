import numpy as np
import matplotlib.pyplot as plt
import sys, torch
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








def evaluate_model(model, eval_xyz=True, visualize=False):

    dataset_titles = [
    "1 fly below",
    "2 fly above",
    "3 swapping",
    "3 swapping (very fast)",
    "all 4 in total"
    ]

    dataset_paths = [
        find_file_with_substring("raw_data_1_flybelow_200Hz_80_005_len34951ts_51_iterations_testset.npz"),
        find_file_with_substring("raw_data_2_flyabove_200Hz_80_005_len32956ts_46_iterations_testset.npz"),
        find_file_with_substring("raw_data_3_swapping_200Hz_80_005_len29736ts_52_iterations_testset.npz"),
        find_file_with_substring("raw_data_3_swapping_fast_200Hz_80_005_len20802ts_67_iterations_testset.npz"),
    ]

    predictions_x = []
    predictions_y = []
    predictions_z = []

    truths_x = []
    truths_y = []
    truths_z = []

    time = []

    dataset_lenghts = []

    for idx, path_idx in enumerate([[0],[1],[2],[3]]):

        # load states to validate on
        uav_1_states = []
        uav_2_states = []
        time_seq = []

        for path in np.array(dataset_paths)[path_idx]:
            
            data = np.load(path)
            uav_1_state, uav_2_state = data['uav_1_states'][:-1], data['uav_2_states'][:-1]

            uav_1_states.extend(uav_1_state)
            uav_2_states.extend(uav_2_state)
            time_seq.extend(np.arange(0,len(uav_1_state)))

        #print("dataset length", len(uav_1_states))

        dataset_lenghts.append(len(uav_1_states))

        rel_state_vector_list = np.array(uav_1_states) - np.array(uav_2_states)
        _, overlap_indices, overlaps = extract_data(uav_1_states, uav_2_states, time_seq)
        u2_x_dw_forces, u2_y_dw_forces, u2_z_dw_forces = extract_dw_forces(uav_2_states)

        prediction = model.evaluate(uav_1_states, uav_2_states)

        #print(prediction[:4])

        if eval_xyz:
            predictions_x.append(prediction[:,0])
            predictions_y.append(prediction[:,1])
            predictions_z.append(prediction[:,2])
            truths_x.append(u2_x_dw_forces)
            truths_y.append(u2_y_dw_forces)
            truths_z.append(u2_z_dw_forces)

        
        else:
            predictions_z.append(prediction)
            truths_z.append(u2_z_dw_forces)

        time.append(time_seq)





        
    
   

    
    #plot_array_with_segments(fig, time_seq, smoothed_u2_z_forces, color="blue", roll=roll_iterations, label="UAV total Z-forces")
    #plot_array_with_segments(fig, time_seq, u2_thrusts,  color="orange", roll=roll_iterations, label="controller z-forces")


    #plot_array_with_segments(fig, np.array(time_seq)[overlaps], u2_z_dw_forces[overlaps], color="magenta", roll=roll_iterations, label="downwash disturbance forces", overlaps=overlaps)
    
    
    all_results = []
    all_results_normalized = []
    
    for idx, pred_z_dw in enumerate(predictions_z):
        

        
        #fig = plt.subplot()
        if visualize:
            fig = plt.subplot()
            #fig.plot(u2_z_forces, label="UAV's z-axis forces") # unsmoothed
            plot_array_with_segments(fig, np.array(time[idx]), truths_z[idx], color="magenta", roll=False, label="downwash disturbance forces", overlaps=overlaps)

            plot_array_with_segments(fig, np.array(time[idx]), pred_z_dw, roll=False)
            plt.ylabel("Force [N]")
            plt.xlabel("time [s]")
            plt.grid()
            plt.title("Actual bottom UAV Z-forces, controller's thrust, and residual downwash force")
            plt.legend()
            plt.show()
        #plot_array_with_segments(fig, np.array(time_seq)[overlaps], prediction[:,1][overlaps], roll=roll_iterations, color=colors[idx], label=labels[idx])

        #compute_rmse(prediction[:,2][overlaps], u2_z_dw_forces[overlaps],label=labels[idx])

        # Compute metrics
        #metrics = compute_metrics(u2_z_dw_forces[overlaps], prediction[:, 2][overlaps]) # in overlap regions only

        #print(truths_z[idx].shape, pred_z_dw.shape)
        metrics = compute_metrics(truths_z[idx], pred_z_dw) # everywhere

        #metrics = compute_metrics(u2_y_dw_forces[overlaps], prediction[:, 1][overlaps])


        metrics["Label"] = dataset_titles[idx]
        all_results.append(metrics)

    


    # Compute average metrics
    average_metrics = {
        "Label": "Average",
        "RMSE": np.mean([res["RMSE"] for res in all_results]),
        "NRMSE (mean)": np.mean([res["NRMSE (mean)"] for res in all_results]),
        "MAE": np.mean([res["MAE"] for res in all_results]),
        "R2 Score": np.mean([res["R2 Score"] for res in all_results])
    }
    
    
    headers = ["Dataset", "RMSE", "NRMSE (mean)", "MAE", "R2 Score"]
    all_results.append(average_metrics)

    table = [[res["Label"], res["RMSE"], res["NRMSE (mean)"], res["MAE"], res["R2 Score"]] for res in all_results]
    print(tabulate(table, headers=headers, tablefmt="grid"))

    


    



model_path = find_file_with_substring("adaptive-partition20")

model = DWPredictor()
model.load_state_dict(torch.load(model_path, weights_only=True))

evaluate_model(model, visualize=False)