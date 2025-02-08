import os, itertools, subprocess

# Define hyperparameter values to test
learning_rates = [1e-3, 3e-4, 1e-4]
epochs = [50, 100]
batch_sizes = [32, 64]

# Generate all combinations of hyperparameters
hyperparam_combinations = list(itertools.product(learning_rates, epochs, batch_sizes))

# Run training script for each combination
for lr, n_epochs, batch_size in hyperparam_combinations:
    print(f"Running training with lr={lr}, epochs={n_epochs}, batch_size={batch_size}")

    # Run training script as a subprocess
    subprocess.run([
        "python", "train_shallow_batched.py",
        "--lr", str(lr),
        "--epochs", str(n_epochs),
        "--batch_size", str(batch_size)
    ])
