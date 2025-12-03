import torch
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import json
import os

from data import sample_data
from models import ScoreNet, dsm_loss, score_matching_loss
from viz import plot_results, plot_langevin_samples

def save_weights_to_json(model, filename):
    weights = []
    for layer in model.net:
        if isinstance(layer, torch.nn.Linear):
            layer_data = {
                'weight': layer.weight.detach().cpu().numpy().tolist(),
                'bias': layer.bias.detach().cpu().numpy().tolist()
            }
            weights.append(layer_data)
    
    with open(filename, 'w') as f:
        json.dump(weights, f)
    print(f"Saved weights to {filename}")

def train(dist_type, n_samples=1000, n_epochs=5000, lr=1e-3, sigma=0.1):
    print(f"Training for {dist_type}...")
    
    # Generate data
    samples = sample_data(dist_type, n_samples)
    dataset = torch.from_numpy(samples).float()
    
    # Setup model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ScoreNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    dataset = dataset.to(device)
    
    # Training loop
    pbar = tqdm(range(n_epochs))
    for epoch in pbar:
        optimizer.zero_grad()
        loss = score_matching_loss(model, dataset)
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            pbar.set_description(f"Loss: {loss.item():.4f}")
            
    return model, samples

def langevin_dynamics(model, n_samples=1000, n_steps=1000, step_size=0.01, device='cpu'):
    model.eval()
    # Start from random noise
    x = torch.randn(n_samples, 2).to(device)
    
    for _ in range(n_steps):
        z = torch.randn_like(x)
        with torch.no_grad():
            score = model(x)
        # Langevin update: x_{t+1} = x_t + score * dt + sqrt(2*dt) * z
        # We assume step_size is dt
        x = x + step_size * score + np.sqrt(2 * step_size) * z
        
    return x.cpu().numpy()

def main():
    distributions = ['gaussian', 'mixture', 'ring', 'swiss_roll']
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Hyperparameters
    # Sigma needs to be chosen carefully. 
    # Too small -> overfitting/spiky. Too large -> blurry.
    sigma = 0.3 
    lr = 0.005
    n_epochs = 1000 # Reduced for speed
    
    for dist in distributions:
        # Train
        model, samples = train(dist, n_samples=1000, n_epochs=n_epochs, lr=lr, sigma=sigma)
        
        # Save weights
        save_weights_to_json(model, f'weights_{dist}.json')

        # Visualize Score Field
        plot_results(model, samples, dist, device=device)
        
        # Langevin Dynamics
        print(f"Running Langevin dynamics for {dist}...")
        generated_samples = langevin_dynamics(model, n_samples=1000, n_steps=1000, step_size=0.01, device=device)
        plot_langevin_samples(samples, generated_samples, dist)
        
    print("Done! Check the results_*.png, langevin_*.png, and weights_*.json files.")

if __name__ == "__main__":
    main()
