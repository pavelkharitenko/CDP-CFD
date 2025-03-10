import sys, torch
sys.path.append('../../utils/')
from utils import *
import json
sys.path.append('../../observers/')

from neuralswarm.model import NeuralSwarmPredictor
from ndp.model import DWPredictor
from SO2.model import ShallowEquivariantPredictor
from agile.model import AgileShallowPredictor, AgilePredictor
from NNARX.model import NNARXModel
from analytical.model import AnalyticalPredictor, AnalyticalPredictorVectorized
import numpy as np
import matplotlib.pyplot as plt





#evaluate predictors 

"""
# store evaluation results in json:
predictors = [
    # Agile
    [AgilePredictor(output_dim=3), ["agile_trained", "agile_runs"], ".pth", True],
    # NDP
    [DWPredictor(), ["ndp_trained", "ndp_runs"], ".pth", True],
    # SO(2)
    [ShallowEquivariantPredictor(), ["SO2_trained", "SO2_runs"], ".pth", True],
    # NS
    [NeuralSwarmPredictor(), ["ns_trained", "ns_runs"], ".pth", False],
]

# Initialize a dictionary to store results for all predictors
all_results = defaultdict(dict)

# Loop over each predictor
for predictor in predictors:
    model, folder_substrings, file_extension, eval_xyz = predictor
    
    # Evaluate the models in the specified folders
    scenario_stats = evaluate_models_in_folder(model, folder_substrings, file_extension, eval_xyz)
    
    # Store the results in the all_results dictionary
    model_name = model.__class__.__name__  # Get the name of the model class
    all_results[model_name] = scenario_stats

# Print the results for all predictors
for model_name, scenario_stats in all_results.items():
    print(f"\nResults for {model_name}:")
    for scenario, stats in scenario_stats.items():
        print(f"{scenario}:")
        print(f"  Mean RMSE: {stats['mean']:.4f}")
        print(f"  Std RMSE: {stats['std']:.4f}")

# Serialize the results to a JSON file
with open("model_evaluation_results.json", "w") as json_file:
    json.dump(all_results, json_file, indent=4)

print("\nResults saved to 'model_evaluation_results.json'")
"""


MyBlue1 = (120/255, 150/255, 200/255)    # Blue (Learnable Queries)
MySand1 = (220/255, 180/255, 120/255)    # Sand / Beige (Bars for layers 3, 6, 9)
MyOrange1 = (230/255, 150/255, 90/255)    # Orange (Backbone)
MyGreen1 = (170/255, 220/255, 150/255)    # Green (Pixel Decoder)
MyPurple1 = (200/255, 170/255, 220/255)   # Purple (Mask layers inside Pixel Decoder)

# Define custom colors using RGB values (Second Set)
MyGray = (150/255, 150/255, 150/255)      # Gray (BiT line and legend)
MyBlue2 = (100/255, 160/255, 230/255)     # Blue (ViT-B/32)
MyPink = (200/255, 120/255, 160/255)      # Pink (ViT-B/16)
MyGreen2 = (120/255, 180/255, 140/255)    # Green (ViT-L/32)
MyLightBlue = (140/255, 200/255, 240/255) # Light Blue (ViT-L/16)
MyOrange2 = (210/255, 140/255, 80/255)    # Orange (ViT-H/14)

# Assign colors to each model
colors = {
    'AgilePredictor': MyBlue2,
    'DWPredictor': MySand1,
    'ShallowEquivariantPredictor': MyPink,
    'NeuralSwarmPredictor': MyGreen2
}

# Define custom display names for models
model_display_names = {
    'AgilePredictor': 'Agile (Proposed)',
    'DWPredictor': 'NDP',
    'ShallowEquivariantPredictor': 'SO(2)-Equiv.',
    'NeuralSwarmPredictor': 'Neural-Swarm'
}

# Define custom display names for scenarios
scenario_display_names = {
    "1 fly below": "Fly Below",
    "2 fly above": "Fly Above",
    "3 swapping": "Swapping",
    "3 swapping (very fast)": "Swapping \n(Fast)"
}

# Load results from JSON file
with open("model_evaluation_results.json", "r") as json_file:
    all_results = json.load(json_file)

# Define scenarios and x-axis values
scenarios = ["1 fly below", "2 fly above", "3 swapping", "3 swapping (very fast)"]
x_values = np.arange(len(scenarios))  # [0, 1, 2, 3]

# Plotting
plt.figure(figsize=(3.5, 2.25))  # Adjusted size for IEEE 2-column paper

# Loop through each model in the results
for model_name, scenario_stats in all_results.items():
    # Extract mean and std RMSE for each scenario
    mean_rmses = [scenario_stats[scenario]["mean"] for scenario in scenarios]
    std_rmses = [scenario_stats[scenario]["std"] for scenario in scenarios]
    
    # Get the custom display name for the model
    display_name = model_display_names.get(model_name, model_name)
    
    # Plot the mean RMSE with thicker and more visible error bars
    plt.errorbar(
        x_values, mean_rmses, yerr=std_rmses, marker='s', markersize=5, capsize=5, capthick=1, elinewidth=1,
        label=display_name, color=colors.get(model_name, MyGray)
    )

# Customize the plot
plt.xticks(x_values, [scenario_display_names[scenario] for scenario in scenarios], fontsize=8)  # Set x-axis labels to custom scenario names
#plt.yticks(fontsize=8)  # Adjust y-axis tick font size
#plt.xlabel("Scenario")
plt.ylabel("RMSE Mean and Std.")
plt.legend(loc='upper left')  # Move legend outside the plot
plt.grid(True, linestyle='--', linewidth=0.5)  # Add a light grid

# Adjust layout to make room for the legend
plt.tight_layout()

# Save the figure in high resolution for publication
plt.savefig("mean_rmse_ieee.pdf", format="pdf", bbox_inches="tight", dpi=300)

# Show plot
plt.show()
exit(0)




# Define custom colors using RGB values (First Set)
MyBlue1 = (120/255, 150/255, 200/255)    # Blue (Learnable Queries)
MySand1 = (220/255, 180/255, 120/255)    # Sand / Beige (Bars for layers 3, 6, 9)
MyOrange1 = (230/255, 150/255, 90/255)    # Orange (Backbone)
MyGreen1 = (170/255, 220/255, 150/255)    # Green (Pixel Decoder)
MyPurple1 = (200/255, 170/255, 220/255)   # Purple (Mask layers inside Pixel Decoder)

# Define custom colors using RGB values (Second Set)
MyGray = (150/255, 150/255, 150/255)      # Gray (BiT line and legend)
MyBlue2 = (100/255, 160/255, 230/255)     # Blue (ViT-B/32)
MyPink = (200/255, 120/255, 160/255)      # Pink (ViT-B/16)
MyGreen2 = (120/255, 180/255, 140/255)    # Green (ViT-L/32)
MyLightBlue = (140/255, 200/255, 240/255) # Light Blue (ViT-L/16)
MyOrange2 = (210/255, 140/255, 80/255)    # Orange (ViT-H/14)

# Example data as numpy arrays
# Each array contains tuples of (mean, variance) for 10M, 30M, 100M, 300M
data = {
    # Original models (First Set)
    'ViT-B/32 (Set 1)': np.array([(30, 3), (40, 2), (50, 2), (60, 1.5)]),
    'ViT-L/16 (Set 1)': np.array([(28, 2.5), (38, 2), (48, 1.5), (58, 1.2)]),
    'ResNet50 (BiT) (Set 1)': np.array([(35, 4), (45, 3), (50, 2), (52, 1)]),
    'ResNet152 (BiT) (Set 1)': np.array([(40, 5), (50, 4), (60, 3), (65, 2)]),

    # New models (Second Set)
    'ViT-B/32 (Set 2)': np.array([(32, 2.8), (42, 1.8), (52, 1.8), (62, 1.3)]),
    'ViT-B/16 (Set 2)': np.array([(34, 2.6), (44, 1.7), (54, 1.6), (64, 1.2)]),
    'ViT-L/32 (Set 2)': np.array([(36, 2.4), (46, 1.6), (56, 1.4), (66, 1.1)]),
    'ViT-L/16 (Set 2)': np.array([(38, 2.2), (48, 1.5), (58, 1.3), (68, 1.0)]),
    'ViT-H/14 (Set 2)': np.array([(40, 2.0), (50, 1.4), (60, 1.2), (70, 0.9)])
}

# Assign colors to each model
colors = {
    # Original models (First Set)
    'ViT-B/32 (Set 1)': MyBlue1,
    'ViT-L/16 (Set 1)': MySand1,
    'ResNet50 (BiT) (Set 1)': MyOrange1,
    'ResNet152 (BiT) (Set 1)': MyGreen1,

    # New models (Second Set)
    'ViT-B/32 (Set 2)': MyBlue2,
    'ViT-B/16 (Set 2)': MyPink,
    'ViT-L/32 (Set 2)': MyGreen2,
    'ViT-L/16 (Set 2)': MyLightBlue,
    'ViT-H/14 (Set 2)': MyOrange2
}

# X-axis values
x_values = np.array([10, 30, 100, 300])

# Plotting
plt.figure(figsize=(14, 8))

for label, values in data.items():
    means = values[:, 0]
    variances = values[:, 1]
    plt.errorbar(x_values, means, yerr=variances, label=label, marker='o', capsize=5, color=colors[label])

# Setting log scale for x-axis
plt.xscale('log')

# Setting x-ticks and labels
plt.xticks(x_values, ['10M', '30M', '100M', '300M'])

# Labels and title
plt.xlabel('Number of JFT pre-training samples')
plt.ylabel('Linear 5-shot ImageNet Top1 [%]')
plt.title('Comparison of Different Models (Original and New Colors)')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))  # Move legend outside the plot
plt.grid(True)

# Setting y-axis limits
plt.ylim(20, 75)

# Adjust layout to make room for the legend
plt.tight_layout()

# Show plot
plt.show()