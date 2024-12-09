import numpy as np
import pysindy as ps
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from dataset import *
from sklearn.preprocessing import StandardScaler

# 1. Load or Generate State Vector and Disturbance Data
# Replace with your actual data
# X: state vectors, shape (n_samples, n_states)
# t: time vector, evenly spaced or otherwise

from scipy.signal import savgol_filter

random.seed(2)

def extract_data_sindy(dataset, normalize=True):
    # Assume dataset.x_arr and dataset.y_arr are already loaded
    states = dataset.x_arr  # Shape: (n_train, 14)
    forces = dataset.y_arr

    #states = dataset.x_arr[:,[0,1,4,5,6,7,8,9,10,11,12,13]]


    if normalize:
        scaler = StandardScaler()
        states = scaler.fit_transform(states)  # Scale the input data
        forces = scaler.fit_transform(forces.reshape(-1,1))  # Scale the input data

    

    threshold = 0.1
    states, forces, timestamp_lists, split_times = split_sequences_by_threshold(states, forces, threshold)


    filtered_sequences = []
    filtered_sequences2 = []
    filtered_timestamps = []

    for seq, forces, t_seq in zip(states, forces, timestamp_lists):
        if seq.shape[0] > 1:  # Check if the sequence has more than 1 data point
            filtered_sequences.append(seq)
            filtered_sequences2.append(forces.reshape(-1,1))
            filtered_timestamps.append(t_seq)
            

    #print(f"Filtered sequences: {len(filtered_sequences)} (removed sequences with fewer than 2 points)")
    # Now train the SINDy model on the filtered data
    states = filtered_sequences
    forces = filtered_sequences2

    t = filtered_timestamps

    # create timestamp array for each trajectory
    timestamps = []
    for state_array in states:
        time = 0.0
        n_states = state_array.shape[0]  # Get the number of states (time steps)
        time_array = np.zeros(n_states)
        for i in range(n_states):
            time_array[i] = time
            time += 0.005
        timestamps.append(time_array)


    t = timestamps

    # shuffle
    combined = list(zip(states, forces, t))

    # Shuffle the combined list
    random.shuffle(combined)

    # Unzip the shuffled list back into individual lists
    states, forces, t = zip(*combined)


    return states, forces, t





# training datasets
dataset = AgileContinousDataset([
    #r"C:\Users\admin\Desktop\IDP\CDP-CFD\arcs\code\data_collection\agile_manuevers\1_flybelow\speeds_05_20\raw_data_1_flybelow_200Hz_80_005_len2176ts_5_iterations.npz",
    r"C:\Users\admin\Desktop\IDP\CDP-CFD\arcs\code\data_collection\agile_manuevers\1_flybelow\speeds_05_20\raw_data_1_flybelow_200Hz_80_005_len68899ts_103_iterations.npz",
    #r"C:\Users\admin\Desktop\IDP\CDP-CFD\arcs\code\data_collection\agile_manuevers\2_flyabove\speeds_05_20\raw_data_2_flyabove_200Hz_80_005_len7636ts_12_iterations.npz",
    #r"C:\Users\admin\Desktop\IDP\CDP-CFD\arcs\code\data_collection\agile_manuevers\2_flyabove\speeds_05_20\raw_data_2_flyabove_200Hz_80_005_len65911ts_91_iterations.npz",
    #r"C:\Users\admin\Desktop\IDP\CDP-CFD\arcs\code\data_collection\agile_manuevers\3_swapping\speeds_05_20\raw_data_3_swapping_200Hz_80_005_len60694ts_100_iterations.npz",
    #r"C:\Users\admin\Desktop\IDP\CDP-CFD\arcs\code\data_collection\agile_manuevers\3_swapping\speeds_20_40\raw_data_3_swapping_fast_200Hz_80_005_len56629ts_173_iterations.npz",

    #r"C:\Users\admin\Desktop\IDP\CDP-CFD\arcs\code\data_collection\agile_manuevers\1_flybelow\speeds_05_20\raw_data_1_flybelow_200Hz_80_005_len34951ts_51_iterations_testset.npz",
    #r"C:\Users\admin\Desktop\IDP\CDP-CFD\arcs\code\data_collection\agile_manuevers\3_swapping\raw_data_3_swapping_fast_200Hz_80_005_len20802ts_67_iterations_testset.npz",
]) 

# training datasets
dataset_eval = AgileContinousDataset([
    r"C:\Users\admin\Desktop\IDP\CDP-CFD\arcs\code\data_collection\agile_manuevers\1_flybelow\speeds_05_20\raw_data_1_flybelow_200Hz_80_005_len34951ts_51_iterations_testset.npz",
    #r"C:\Users\admin\Desktop\IDP\CDP-CFD\arcs\code\data_collection\agile_manuevers\2_flyabove\speeds_05_20\raw_data_2_flyabove_200Hz_80_005_len32956ts_46_iterations_testset.npz",
    #r"C:\Users\admin\Desktop\IDP\CDP-CFD\arcs\code\data_collection\agile_manuevers\3_swapping\speeds_05_20\raw_data_3_swapping_200Hz_80_005_len29736ts_52_iterations_testset.npz",
    #r"C:\Users\admin\Desktop\IDP\CDP-CFD\arcs\code\data_collection\agile_manuevers\3_swapping\speeds_20_40\raw_data_3_swapping_fast_200Hz_80_005_len20802ts_67_iterations_testset.npz",
    #r"C:\Users\admin\Desktop\IDP\CDP-CFD\arcs\code\data_collection\agile_manuevers\3_swapping\raw_data_3_swapping_fast_200Hz_80_005_len20802ts_67_iterations_testset.npz",
]) 


states, forces, t = extract_data_sindy(dataset)
states_eval, forces_eval, t_eval = extract_data_sindy(dataset_eval)









# # Time vector (assuming the data is sampled at uniform time intervals)
# n_samples = len(states)
# t = np.linspace(0.0, 0.02*n_samples, n_samples)
# # 1. Plot the states and forces
# plt.figure(figsize=(12, 6))
# plt.subplot(2, 1, 1)
# plt.plot(t, states[:, :3])  # Plot the first three state variables
# plt.title("Sample UAV States (Position, Velocity, etc.)")
# plt.xlabel("Time")
# plt.ylabel("State Values")

# plt.subplot(2, 1, 2)
# plt.plot(t, forces)  # Plot the disturbance force
# plt.title("Aerodynamic Disturbance Forces")
# plt.xlabel("Time")
# plt.ylabel("Forces")
# plt.tight_layout()
#plt.show()

# 2. Compute Time Derivatives (Numerical Differentiation)
# Use finite differences to compute the derivatives of states and forces
#state_derivative = ps.FiniteDifference(order=2)._differentiate(states, t)
#force_derivative = ps.FiniteDifference(order=2)._differentiate(forces, t)

# 3. Train the SINDy Model to Learn Disturbance Dynamics
# Setup the SINDy model with polynomial library for features and sparse regression optimizer





custom_functions = [
    lambda x: np.exp(x),                  # Exponential
    lambda x: np.log(np.abs(x) + 1e-6),   # Logarithm (avoid log(0))
    lambda x: np.sqrt(np.abs(x)),         # Square root (fractional exponent 1/2)
    lambda x: np.power(np.abs(x), 1.5),   # Fractional exponent 3/2
]

# Define the names for the custom library features
custom_function_names = [ 
    lambda x: "exp(" + x + ")",
    lambda x: "log(|" + x + "|)",
    lambda x: "sqrt(|" + x + "|)",             # Square root
    lambda x: "|" + x + "|" + "^(3/2)",                # Fractional exponent 3/2
    #lambda x: "sin(" + x + ")*cos(" + x + ")"
]

# Wrap the custom functions into a CustomLibrary
custom_lib = ps.CustomLibrary(
    library_functions=custom_functions,
    function_names=custom_function_names
)

# Combine all libraries: Polynomial, Fourier, and Custom
poly_lib = ps.PolynomialLibrary(degree=2, include_interaction=True)
fourier_lib = ps.FourierLibrary(n_frequencies=2)
combined_lib = poly_lib 
combined_lib += custom_lib 
combined_lib += fourier_lib




smoother_kws = {"window_length": 21, "polyorder":2}
derivative = ps.SmoothedFiniteDifference(order=2, drop_endpoints=True, smoother_kws=smoother_kws)

# model = ps.SINDy(
#     #feature_names=[f"x{i+1}" for i in range(15)],
#     differentiation_method=derivative,
#     optimizer=ps.STLSQ(threshold=0.05, max_iter=100),  # Lower threshold
#     feature_library=ps.PolynomialLibrary(degree=2),  # Higher degree
# )

model = ps.SINDy(
    #feature_names=[f"x{i+1}" for i in range(15)],
    #differentiation_method=derivative,
    optimizer=ps.STLSQ(threshold=0.9, max_iter=2000, alpha=0.9),  # Lower threshold
    #feature_library=ps.PolynomialLibrary(degree=2),  
    feature_library=combined_lib
)



print("Training the SINDy model to learn disturbance dynamics...")
model.fit(x=forces, t=t, 
          
          #x_dot=force_derivative, 
          u=states, multiple_trajectories=True)
model.print()  # Print the learned model


# 4. Predict Disturbance Forces (y) Given States (x)
# Predict the disturbance forces using the trained SINDy model





#force_predicted_der = model.predict(np.concatenate(forces), u=np.concatenate(states))
force_predicted_der = model.predict(forces, u=states, multiple_trajectories=True)
force_predicted_der = np.concatenate(force_predicted_der)
force_derivative = ps.FiniteDifference(order=2)._differentiate(np.concatenate(forces), np.concatenate(t))
force_derivative = np.squeeze(force_derivative)


# 5. Plot the True vs Predicted Forces
plt.figure(figsize=(10, 6))
plt.plot(force_derivative, label="True Force derivative", linewidth=2)
window_length = 21
print("Window length", window_length)
force_derivative_smooth = savgol_filter(force_derivative, window_length, 1)
plt.plot(force_derivative_smooth, label="Smoothed Force Derivative", linewidth=2, color="orange")
plt.plot(force_predicted_der, '--', label="Predicted Force derivative (SINDy)", linewidth=2, color="pink")
rmse = np.sqrt(mean_squared_error(force_derivative_smooth, force_predicted_der))
print(f"RMSE between Smoothed True Force Derivative and Predicted: {rmse:.4f}")

plt.xlabel("Time Step")
plt.ylabel("Force Derivative")
plt.title(f"Comparison of Force Derivatives (RMSE: {rmse:.4f})")
plt.legend()
plt.show()


# 6. Evaluate Model Performance (Optional)
# Compute the R^2 score or any other evaluation metric


print("----------- Evaluating derivative -------------")

mse = mean_squared_error(force_derivative_smooth, force_predicted_der)
rmse = np.sqrt(mean_squared_error(force_derivative_smooth, force_predicted_der))

r2 = r2_score(force_derivative_smooth, force_predicted_der)
test_score = model.score(x=forces, u=states, t=t, multiple_trajectories=True)

print(f"Mean Squared Error: {mse:.4f}")
print(f"Root Mean Squared Error: {rmse:.4f}")

print(f"R^2 Score on smoothed trainining data: {r2:.4f}")
print(f"pysindys R^2 Score on training data: {test_score:.4f}")
print("----------- Evaluating actual x(t) -------------")

start_id = 100
end_id = -50
step_id = 20
iteration = 3
simulated_states = model.simulate(forces_eval[iteration][start_id], 
                                  t_eval[iteration][start_id:end_id:step_id], 
                                  u=states_eval[iteration][start_id:end_id:step_id])

plt.plot(t_eval[iteration][start_id:end_id:step_id], forces_eval[iteration][start_id:end_id:step_id], label="True forces", linewidth=2)
plt.plot(t_eval[iteration][start_id:end_id:step_id][1:],simulated_states, label="Simulated forces using SINDy", linewidth=2)
plt.title("True vs Predicted Aerodynamic Disturbance Forces")
plt.xlabel("Time")
plt.ylabel("Forces")
plt.legend()
plt.show()



forces_to_be_predicted = forces_eval[iteration][start_id:end_id:step_id][1:]
forces_to_be_predicted = np.squeeze(forces_to_be_predicted)
forces_to_be_predicted = smooth_with_savgol(forces_to_be_predicted, window_size=5, poly_order=1)

rmse_sampled = np.sqrt(mean_squared_error(forces_to_be_predicted, simulated_states))

print(f"Sampled RMSE: {rmse_sampled:.3f}")

force_predicted_der_eval = model.predict(forces_eval, u=states_eval, multiple_trajectories=True)
force_predicted_der_eval = np.concatenate(force_predicted_der_eval)
force_derivative_eval = ps.FiniteDifference(order=2)._differentiate(np.concatenate(forces_eval), np.concatenate(t_eval))
force_derivative_eval = np.squeeze(force_derivative_eval)

window_length = 21
print("Window length", window_length)
force_derivative_smooth_eval = savgol_filter(force_derivative_eval, window_length, 1)

mse = mean_squared_error(force_derivative_smooth_eval, force_predicted_der_eval)
rmse = np.sqrt(mean_squared_error(force_derivative_smooth_eval, force_predicted_der_eval))

r2 = r2_score(force_derivative_smooth_eval, force_predicted_der_eval)
print(f"R^2 Score on smoothed evaluation data: {r2:.4f}")

r2_score_forces = model.score(x=forces_eval, t=t_eval, u=states_eval, multiple_trajectories=True)
print(f"pysindys R^2 Score on evaluation data: {r2_score_forces:.4f}")




