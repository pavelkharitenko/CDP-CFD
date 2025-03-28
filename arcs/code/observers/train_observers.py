import subprocess, sys
#from utils import *
#sys.path.append('../utils/')


learning_rates = [1e-3, 2e-4, 5e-4]
batch_sizes = [64]
epochs_list = [300]
#seeds = [123, 456]
seeds = [789, 101112, 131415]

SNs = [None]

# Path to your training script
train_script = "train_batched.py"

# Loop over all combinations of hyperparameters and seeds
for lr in learning_rates:
    for bs in batch_sizes:
        for epochs in epochs_list:
            for seed in seeds:
                for sn in SNs:
                    print(f"Running experiment with lr={lr}, batch_size={bs}, epochs={epochs}, seed={seed}, sn={sn}")
                    
                    # Construct the command to run the training script
                    command = [
                        "python", train_script,
                        "--lr", str(lr),
                        "--batch_size", str(bs),
                        "--epochs", str(epochs),
                        "--seed", str(seed),
                        "--save_model", "True",
                        #"--sn_gamma", str(sn),
                    ]
                    
                    # Run the command
                    subprocess.run(command)