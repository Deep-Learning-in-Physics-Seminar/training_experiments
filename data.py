import numpy as np

def sample_gaussian(n_samples):
    return np.random.normal(0, 1, (n_samples, 2))

def sample_mixture(n_samples):
    # 0.5 prob for each component
    indices = np.random.choice([0, 1], size=n_samples, p=[0.5, 0.5])
    samples = np.zeros((n_samples, 2))
    
    # Component 0: mu=(-2.5, -2.5), sigma=0.8
    mask0 = (indices == 0)
    samples[mask0] = np.random.normal(-2.5, 0.8, (mask0.sum(), 2))
    
    # Component 1: mu=(2.5, 2.5), sigma=0.8
    mask1 = (indices == 1)
    samples[mask1] = np.random.normal(2.5, 0.8, (mask1.sum(), 2))
    
    return samples

def sample_ring(n_samples):
    n_components = 8
    radius = 3.5
    sigma = 0.5
    
    # Uniformly choose from 8 components
    indices = np.random.choice(n_components, size=n_samples)
    samples = np.zeros((n_samples, 2))
    
    for i in range(n_components):
        mask = (indices == i)
        angle = (i / n_components) * 2 * np.pi
        mu_x = np.cos(angle) * radius
        mu_y = np.sin(angle) * radius
        
        samples[mask] = np.random.normal([mu_x, mu_y], sigma, (mask.sum(), 2))
        
    return samples

def sample_swiss_roll(n_samples):
    n_components = 15
    
    # Uniformly choose from 15 components
    indices = np.random.choice(n_components, size=n_samples)
    samples = np.zeros((n_samples, 2))
    
    for i in range(n_components):
        mask = (indices == i)
        
        # t goes from 1.5*pi to 4.5*pi (1 + 2*i/15)
        t = 1.5 * np.pi * (1 + 2 * i / n_components)
        r = 0.5 + 0.3 * t
        
        # Scaling factor 0.4 from JS code
        mu_x = r * np.cos(t) * 0.4
        mu_y = r * np.sin(t) * 0.4
        sigma = 0.35
        
        samples[mask] = np.random.normal([mu_x, mu_y], sigma, (mask.sum(), 2))
        
    return samples

def sample_data(dist_type, n_samples=1000):
    if dist_type == 'gaussian':
        return sample_gaussian(n_samples)
    elif dist_type == 'mixture':
        return sample_mixture(n_samples)
    elif dist_type == 'ring':
        return sample_ring(n_samples)
    elif dist_type == 'swiss_roll':
        return sample_swiss_roll(n_samples)
    else:
        raise ValueError(f"Unknown distribution type: {dist_type}")
