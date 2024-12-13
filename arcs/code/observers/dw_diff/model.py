import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
import torch.nn.functional as F
from dataset import *
from scipy.signal import savgol_filter

device = "cuda" if torch.cuda.is_available() else "cpu"


def extract_data_sindy(dataset, normalize=True):
    # Assume dataset.x_arr and dataset.y_arr are already loaded
    states = dataset.x_arr  # Shape: (n_train, 14)
    forces = dataset.y_arr


    return states, forces



device = "cuda" if torch.cuda.is_available() else "cpu"


def extract_data_sindy(dataset, normalize=True):
    # Assume dataset.x_arr and dataset.y_arr are already loaded
    states = dataset.x_arr  # Shape: (n_train, 14)
    forces = dataset.y_arr


    return states, forces







# Diffusion noise scheduler
class DiffusionScheduler:
    def __init__(self, timesteps=400, beta_start=1e-4, beta_end=0.02):
        self.timesteps = timesteps
        self.beta = torch.linspace(beta_start, beta_end, timesteps, device="cuda")  # Noise schedule
        self.alpha = 1.0 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)  # Cumulative product of alphas

    def get_noise_level(self, t):
        return self.alpha_hat[t]

    def add_noise(self, y, t):
        """
        Adds Gaussian noise to the target forces y based on timestep t.
        """
        noise = torch.randn_like(y)
        alpha_hat_t = self.alpha_hat[t].view(-1, 1).to(y.device)
        noisy_y = torch.sqrt(alpha_hat_t) * y + torch.sqrt(1 - alpha_hat_t) * noise
        return noisy_y, noise



class ConditionalDenoiser(nn.Module):
    def __init__(self, state_dim, force_dim, hidden_dim=64):
        super(ConditionalDenoiser, self).__init__()
        self.state_embed = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        self.noise_predictor = nn.Sequential(
            nn.Linear(hidden_dim + force_dim + 1, hidden_dim),  # +1 for timestep t
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, force_dim)
        )

    def forward(self, noisy_y, x, t):
        """
        noisy_y: Noisy forces (B, force_dim)
        x: States (B, state_dim)
        t: Timestep (scalar or tensor)
        """
        x_embed = self.state_embed(x)  # Embed the states
        t_embed = t / 400.0  # Normalize timestep
        t_embed = t_embed.unsqueeze(1)  # Expand for concatenation
        combined = torch.cat([noisy_y, x_embed, t_embed], dim=-1)
        predicted_noise = self.noise_predictor(combined)
        return predicted_noise

def train_diffusion(model, scheduler, dataloader, optimizer, epochs=20, device="cuda"):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for states, forces in dataloader:
            states, forces = states.to(device), forces.to(device)
            optimizer.zero_grad()

            # Sample a timestep uniformly
            batch_size = states.shape[0]
            t = torch.randint(0, scheduler.timesteps, (batch_size,), device=device)

            # Add noise to forces
            noisy_forces, noise = scheduler.add_noise(forces, t)

            # Predict noise
            predicted_noise = model(noisy_forces, states, t)

            # Loss: MSE between predicted and actual noise
            loss = F.mse_loss(predicted_noise, noise)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(dataloader):.6f}")





@torch.no_grad()
def evaluate_diffusion(model, scheduler, dataloader, device="cuda"):
    model.eval()
    predictions = []
    actuals = []

    for states, forces in dataloader:
        states, forces = states.to(device), forces.to(device)
        
        # Sample forces (predict disturbances) using the trained diffusion model
        sampled_forces = sample_forces(model, scheduler, states, device=device)
        
        predictions.append(sampled_forces.cpu().numpy())
        actuals.append(forces.cpu().numpy())

    # Stack the results for RMSE calculation
    predictions = np.vstack(predictions)
    actuals = np.vstack(actuals)

    # Compute RMSE and R² scores
    mse = mean_squared_error(actuals, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(actuals, predictions)

    print(f"Evaluation Results: RMSE: {rmse:.4f}, R²: {r2:.4f}")
    return actuals, predictions, rmse, r2




@torch.no_grad()
def sample_forces(model, scheduler, states, device="cuda"):
    model.eval()
    batch_size = states.shape[0]
    y = torch.randn((batch_size, 1), device=device)  # Start from pure noise

    for t in reversed(range(scheduler.timesteps)):
        t_tensor = torch.tensor([t] * batch_size, device=device)
        noise_pred = model(y, states, t_tensor)
        alpha = scheduler.alpha[t]
        alpha_hat = scheduler.alpha_hat[t]
        beta = scheduler.beta[t]

        # Update step
        if t > 0:
            noise = torch.randn_like(y)
        else:
            noise = 0

        y = (1 / torch.sqrt(alpha)) * (y - (1 - alpha) / torch.sqrt(1 - alpha_hat) * noise_pred) + torch.sqrt(beta) * noise

    return y



# --- Step 6: Main Script ---
if __name__ == "__main__":
    # Set random seed

    SAVE_MODEL = False

    lags = 15
    n_epochs = 100
    exp_name = init_experiment(f"DWDiff-lags-{str(lags)}-123S")

    random.seed(2)
    torch.manual_seed(2)
    np.random.seed(2)

    dataset = AgileContinousDataset([
        r"C:\Users\admin\Desktop\IDP\CDP-CFD\arcs\code\data_collection\agile_manuevers\1_flybelow\speeds_05_20\raw_data_1_flybelow_200Hz_80_005_len68899ts_103_iterations.npz",
        r"C:\Users\admin\Desktop\IDP\CDP-CFD\arcs\code\data_collection\agile_manuevers\2_flyabove\speeds_05_20\raw_data_2_flyabove_200Hz_80_005_len7636ts_12_iterations.npz",
        #r"C:\Users\admin\Desktop\IDP\CDP-CFD\arcs\code\data_collection\agile_manuevers\2_flyabove\speeds_05_20\raw_data_2_flyabove_200Hz_80_005_len65911ts_91_iterations.npz",
        #r"C:\Users\admin\Desktop\IDP\CDP-CFD\arcs\code\data_collection\agile_manuevers\3_swapping\speeds_05_20\raw_data_3_swapping_200Hz_80_005_len60694ts_100_iterations.npz",
        #r"C:\Users\admin\Desktop\IDP\CDP-CFD\arcs\code\data_collection\agile_manuevers\3_swapping\speeds_20_40\raw_data_3_swapping_fast_200Hz_80_005_len56629ts_173_iterations.npz",
        ])
    
    dataset_eval = AgileContinousDataset([
        r"C:\Users\admin\Desktop\IDP\CDP-CFD\arcs\code\data_collection\agile_manuevers\1_flybelow\speeds_05_20\raw_data_1_flybelow_200Hz_80_005_len34951ts_51_iterations_testset.npz",
        r"C:\Users\admin\Desktop\IDP\CDP-CFD\arcs\code\data_collection\agile_manuevers\2_flyabove\speeds_05_20\raw_data_2_flyabove_200Hz_80_005_len32956ts_46_iterations_testset.npz",
        #r"C:\Users\admin\Desktop\IDP\CDP-CFD\arcs\code\data_collection\agile_manuevers\3_swapping\speeds_05_20\raw_data_3_swapping_200Hz_80_005_len29736ts_52_iterations_testset.npz",
        #r"C:\Users\admin\Desktop\IDP\CDP-CFD\arcs\code\data_collection\agile_manuevers\3_swapping\speeds_20_40\raw_data_3_swapping_fast_200Hz_80_005_len20802ts_67_iterations_testset.npz",
        ])

    # Data preparation

    
    train_data, val_data = train_test_split(dataset, test_size=0.2, shuffle=True)

    batch_size = 64




    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    eval_loader = DataLoader(dataset_eval, batch_size=batch_size, shuffle=False)
    
    
    model = ConditionalDenoiser(state_dim=14, force_dim=1, hidden_dim=64)
    model = model.to(device)

    # Optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    # Train the model
    print("Training diffusion model...")
    scheduler = DiffusionScheduler()
    train_diffusion(model, scheduler, train_loader, optimizer, epochs=n_epochs, device="cuda")



    # Evaluate the model
    print("Evaluating diffusion model...")
    actuals, predictions, rmse, r2 = evaluate_diffusion(model, scheduler, eval_loader)

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(actuals, label="True Forces", linewidth=2)
    plt.plot(predictions, label="Predicted Forces (Diffusion Model)", linewidth=2, linestyle="--")
    plt.xlabel("Time Step")
    plt.ylabel("Force")
    title="True vs Predicted Disturbance Forces"
    plt.title(title)
    plt.legend()
    plt.show()

    exit(0)

    final_model_name = exp_name + str(n_epochs) + "_eps"  + ".pth"
    if SAVE_MODEL:
        print("Training finished, saving...", final_model_name)
        torch.save(model.state_dict(), final_model_name)
