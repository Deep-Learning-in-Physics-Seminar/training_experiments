import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_results(model, samples, dist_type, device='cpu'):
    model.eval()
    
    # Create grid
    x = np.linspace(-5, 5, 30)
    y = np.linspace(-5, 5, 30)
    X, Y = np.meshgrid(x, y)
    grid = np.stack([X.ravel(), Y.ravel()], axis=1)
    grid_tensor = torch.from_numpy(grid).float().to(device)
    
    # Compute scores
    with torch.no_grad():
        scores = model(grid_tensor).cpu().numpy()
    
    U = scores[:, 0].reshape(X.shape)
    V = scores[:, 1].reshape(Y.shape)
    
    # Normalize for better visualization (direction is more important than magnitude for quiver)
    # But magnitude also tells us where the gradients are strong.
    # Let's clip very large values for better plotting
    magnitude = np.sqrt(U**2 + V**2)
    # U = U / (magnitude + 1e-5)
    # V = V / (magnitude + 1e-5)
    
    plt.figure(figsize=(10, 10))
    
    # Plot vector field
    plt.quiver(X, Y, U, V, magnitude, cmap='viridis', scale=50, width=0.005)
    
    # Plot training samples
    plt.scatter(samples[:, 0], samples[:, 1], s=10, alpha=0.6, color='red', label='Training Data')
    
    plt.title(f'Learned Score Field: {dist_type}')
    plt.legend()
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.grid(True, alpha=0.3)
    plt.savefig(f'results_{dist_type}.png')
    plt.close()

def plot_langevin_samples(samples, generated_samples, dist_type):
    plt.figure(figsize=(10, 10))
    plt.scatter(samples[:, 0], samples[:, 1], s=10, alpha=0.5, color='red', label='True Data')
    plt.scatter(generated_samples[:, 0], generated_samples[:, 1], s=10, alpha=0.5, color='blue', label='Generated Data')
    plt.title(f'Langevin Sampling: {dist_type}')
    plt.legend()
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.grid(True, alpha=0.3)
    plt.savefig(f'langevin_{dist_type}.png')
    plt.close()
