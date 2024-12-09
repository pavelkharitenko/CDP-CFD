import numpy as np
import pysindy as ps
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from dataset import *

# 1. Load or Generate State Vector and Disturbance Data
# Replace with your actual data
# X: state vectors, shape (n_samples, n_states)
# t: time vector, evenly spaced or otherwise

from scipy.signal import savgol_filter

# Load the Dataset
# Assuming dataset.x are the states (n, 14) and dataset.y are forces (n, 1)
dataset = AgileContinousDataset([
    #r"C:\Users\admin\Desktop\IDP\CDP-CFD\arcs\code\data_collection\agile_manuevers\1_flybelow\speeds_05_20\raw_data_1_flybelow_200Hz_80_005_len34951ts_51_iterations_testset.npz",
    r"C:\Users\admin\Desktop\IDP\CDP-CFD\arcs\code\data_collection\agile_manuevers\1_flybelow\speeds_05_20\raw_data_1_flybelow_200Hz_80_005_len2176ts_5_iterations.npz",
    #r"C:\Users\admin\Desktop\IDP\CDP-CFD\arcs\code\data_collection\agile_manuevers\1_flybelow\speeds_05_20\raw_data_1_flybelow_200Hz_80_005_len68899ts_103_iterations.npz",
]) 

# Data Splitting
#x_train, x_val, y_train, y_val = train_test_split(dataset.x, dataset.y, train_size=0.75, test_size=0.25, shuffle=True)



# Assume dataset.x_arr and dataset.y_arr are already loaded
states = dataset.x_arr  # Shape: (n_train, 14)
#states = states[:, :12]
forces = dataset.y_arr.reshape(-1, 1)  # Shape: (n_train, 1)

# Time vector (assuming the data is sampled at uniform time intervals)
n_samples = len(states)
d = 3
t = np.linspace(0.0, 0.02*n_samples, n_samples)

# 1. Plot the states and forces
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(t, states[:, :3])  # Plot the first three state variables
plt.title("Sample UAV States (Position, Velocity, etc.)")
plt.xlabel("Time")
plt.ylabel("State Values")

plt.subplot(2, 1, 2)
plt.plot(t, forces)  # Plot the disturbance force
plt.title("Aerodynamic Disturbance Forces")
plt.xlabel("Time")
plt.ylabel("Forces")
plt.tight_layout()
#plt.show()

# 2. Compute Time Derivatives (Numerical Differentiation)
# Use finite differences to compute the derivatives of states and forces
state_derivative = ps.FiniteDifference(order=2)._differentiate(states, t)
force_derivative = ps.FiniteDifference(order=2)._differentiate(forces, t)

# 3. Train the SINDy Model to Learn Disturbance Dynamics
# Setup the SINDy model with polynomial library for features and sparse regression optimizer


model = ps.SINDy(
    feature_names=[f"x{i+1}" for i in range(14)],
    optimizer=ps.STLSQ(threshold=0.005),  # Lower threshold
    feature_library=ps.PolynomialLibrary(degree=2),  # Higher degree
)
# functions = [lambda x : np.exp(x), lambda x,y : np.sin(x+y)]

# lib_custom = ps.CustomLibrary(library_functions=functions)

# lib_fourier = ps.FourierLibrary()

# lib_pol = ps.PolynomialLibrary(degree=4)

# lib_concat = ps.ConcatLibrary([lib_custom, lib_fourier, lib_pol])
# model = ps.SINDy(
#     feature_names=[f"x{i+1}" for i in range(6)],
#     optimizer=ps.STLSQ(threshold=0.0000001),
#     feature_library=lib_concat,
# )

# Fit the model to predict aerodynamic force derivatives
print("Training the SINDy model to learn disturbance dynamics...")
model.fit(states, t=t, x_dot=force_derivative)
model.print()  # Print the learned model

# 4. Predict Disturbance Forces (y) Given States (x)
# Predict the disturbance forces using the trained SINDy model
force_predicted = model.predict(states)

# 5. Plot the True vs Predicted Forces
plt.figure(figsize=(10, 6))
plt.plot(t, forces, label="True Forces", linewidth=2)
plt.plot(t, force_predicted, '--', label="Predicted Forces (SINDy)", linewidth=2)
plt.title("True vs Predicted Aerodynamic Disturbance Forces")
plt.xlabel("Time")
plt.ylabel("Forces")
plt.legend()
plt.show()

# 6. Evaluate Model Performance (Optional)
# Compute the R^2 score or any other evaluation metric
from sklearn.metrics import mean_squared_error, r2_score

mse = mean_squared_error(forces, force_predicted)
r2 = r2_score(forces, force_predicted)

print(f"Mean Squared Error: {mse:.4f}")
print(f"R^2 Score: {r2:.4f}")
