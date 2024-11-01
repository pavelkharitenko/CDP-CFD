import matplotlib.pyplot as plt
import numpy as np

data = np.load("rotor_thrust_map_005_80_rps340_360.npz")
rotor_measurements = np.array(data["rotor_measurements"])
force_averages = np.array(data["force_averages"])


data2 = np.load("rotor_thrust_map_005_80_rps361_400.npz")
rotor_measurements2 = np.array(data2["rotor_measurements"])
force_averages2 = np.array(data2["force_averages"])

data3 = np.load("rotor_thrust_map_005_80_rps361_400_430.npz")
rotor_measurements3 = np.array(data3["rotor_measurements"])
force_averages3 = np.array(data3["force_averages"])



total_meas = np.concatenate((rotor_measurements, rotor_measurements2, rotor_measurements3))
total_forces = np.concatenate((force_averages, force_averages2, force_averages3))


# account gravity force on JFT sensor
total_forces += 29.7733538



# Example data points
x = total_meas
y = total_forces

# Linear fit (1st degree polynomial)
linear_coeffs = np.polyfit(x, y, 1)
linear_fit = np.poly1d(linear_coeffs)
a, b = linear_coeffs  # Linear coefficients

# Quadratic fit (2nd degree polynomial)
quadratic_coeffs = np.polyfit(x, y, 2)
quadratic_fit = np.poly1d(quadratic_coeffs)
c, d, e = quadratic_coeffs  # Quadratic coefficients

# Generate predictions for both fits
y_linear_pred = linear_fit(x)
y_quadratic_pred = quadratic_fit(x)

# Calculate Mean Squared Errors manually
linear_mse = np.mean((y - y_linear_pred) ** 2)
quadratic_mse = np.mean((y - y_quadratic_pred) ** 2)

# Determine the better fit
better_fit = "Linear" if linear_mse < quadratic_mse else "Quadratic"
better_mse = min(linear_mse, quadratic_mse)

# Plot data and fits
fig, axs = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [3, 1]})

# Main plot with data points and fits
axs[0].scatter(x, y, color='black', label='Data Points')
axs[0].plot(x, y_linear_pred, color='blue', label=f'Linear Fit: y = {a:.2f}x + {b:.2f}\nMSE={linear_mse:.4f}')
axs[0].plot(x, y_quadratic_pred, color='red', linestyle='--', label=f'Quadratic Fit: y = {c:.2f}x^2 + {d:.2f}x + {e:.2f}\nMSE={quadratic_mse:.4f}')
axs[0].legend()
axs[0].set_xlabel("RPS of all rotors")
axs[0].set_ylabel("Force in Newton")
axs[0].set_title("Linear and Quadratic Fits to Rps-to-Thrust (res=0.05, max.ref.vel.=80)")
axs[0].text(0.05, 0.95, f"Better fit: {better_fit}", transform=axs[0].transAxes, fontsize=12, verticalalignment='top', bbox=dict(facecolor='lightgray', alpha=0.5))

# Add table with coefficients and errors in a subplot
table_data = [
    ["Model", "Equation", "Coefficients", "MSE"],
    ["Linear", f"y = ax + b", f"a = {a:.2f},\nb = {b:.2f}", f"{linear_mse:.4f}"],
    ["Quadratic", f"y = \ncx^2 + dx + e", f"\nc = {c}, \nd = {d}, \ne = {e}", f"{quadratic_mse:.4f}"],
]
axs[1].axis("off")
table = axs[1].table(cellText=table_data, colLabels=None, cellLoc='center', loc="center")
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(2.0, 4.0)

plt.tight_layout()
plt.show()
