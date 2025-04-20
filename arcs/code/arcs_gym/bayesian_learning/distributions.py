from scipy.stats import norm, entropy
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(seed=0)
mu, sigma = 0, 1  # Mean and standard deviation
num_samples = 100  # Number of random samples

# Generate random samples
samples = norm.rvs(loc=mu, scale=sigma, size=num_samples)


bins = 300


# Create histogram to estimate probabilities
counts, bin_edges = np.histogram(samples, bins=bins, density=True)
probabilities = counts * np.diff(bin_edges)  # Convert to probabilities


# Compute discrete entropy (handle zero probabilities)
entropy_value = entropy(probabilities[probabilities > 0], base=2)
print(f"Empirical entropy (bits): {entropy_value:.4f}")

# Plot histogram
plt.figure(figsize=(8, 6))
plt.hist(samples, bins=bins,density=True, alpha=0.7,  edgecolor='black')
plt.title('Histogram of Random Samples from Gaussian Distribution')
plt.xlabel('Value')
plt.ylabel('Density')
plt.grid(True)
plt.show()


