import sys, torch
sys.path.append('../../utils/')
from utils import *

sys.path.append('../../observers/')

from neuralswarm.model import NeuralSwarmPredictor
from ndp.model import DWPredictor
from SO2.model import ShallowEquivariantPredictor
from agile.model import AgileShallowPredictor, AgilePredictor
from NNARX.model import NNARXModel
from analytical.model import AnalyticalPredictor, AnalyticalPredictorVectorized





#evaluate agile
"""
model_paths = find_files_in_folder_with_substring("agile_trained", ".pth")

for model_path in model_paths:
    print(model_path)
    #model_path = find_file_with_substring(model_path)
    model = AgilePredictor(output_dim=3)
    model.load_state_dict(torch.load(model_path, weights_only=True))

    evaluate_model(model, visualize=False, eval_xyz=True)
"""
#evaluate agile
model_paths = find_files_in_folder_with_substring("agile_trained", ".pth")

for model_path in model_paths:
    print(model_path)
    #model_path = find_file_with_substring(model_path)
    model = AgilePredictor(output_dim=3)
    model.load_state_dict(torch.load(model_path, weights_only=True))


    res_dict = evaluate_model_rmse_per_scenario(model, eval_xyz=True)

    print(res_dict["3 swapping"]["Z-Axis"])

# evaluate ndp
# model_paths = find_files_in_folder_with_substring("ndp", ".pth")

# for model_path in model_paths:
#     print(model_path)
#     #model_path = find_file_with_substring(model_path)
#     model = DWPredictor()
#     model.load_state_dict(torch.load(model_path, weights_only=True))

#     evaluate_model(model, visualize=False, eval_xyz=True)

# evaluate SO(2)-Equiv.
# model_paths = find_files_in_folder_with_substring("SO2", ".pth")

# for model_path in model_paths:
#     print(model_path)
#     #model_path = find_file_with_substring(model_path)
#     model = ShallowEquivariantPredictor()
#     model.load_state_dict(torch.load(model_path, weights_only=True))

#     evaluate_model(model, visualize=False, eval_xyz=True)

# evaluate empirical

# k1s = [4.2] # hover magnitude factor
# k2s = [5.6] # spread factor
# k3s = [0.9] # development length factor

# # Loop over all combinations of hyperparameters and seeds
# for k1 in k1s:
#     for k2 in k2s:
#         for k3 in k3s:
#             print(f"Evaluating AP with k1={k1}, k2={k2}, k3={k3}")
#             ap = AnalyticalPredictorVectorized(k1 = k1, k2 = k2, k3 = k3)
#             evaluate_analytical(ap, visualize=True)


# evaluate normalized agile

# model_paths = find_files_in_folder_with_substring("agile_normalized", ".pth")

# for model_path in model_paths:
#     print(model_path)
#     #model_path = find_file_with_substring(model_path)
#     model = AgilePredictor(output_dim=3)
#     model.load_state_dict(torch.load(model_path, weights_only=True))

#     evaluate_model(model, visualize=False, eval_xyz=True)

# 2.3 mashed vector
# 1.9 urbane access
# 2.0 human fathom

# model_path = find_file_with_substring("adaptive-partition20")
# model = DWPredictor()
# model.load_state_dict(torch.load(model_path, weights_only=True))

# model_path = find_file_with_substring("proper-hardball")
# model = AgilePredictor(output_dim=3)
# model.load_state_dict(torch.load(model_path, weights_only=True))

# evaluate_model(model, visualize=True, eval_xyz=True)