import torch
import torch.nn as nn

class ScoreNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 128),
            nn.Softplus(),
            nn.Linear(128, 128),
            nn.Softplus(),
            nn.Linear(128, 128),
            nn.Softplus(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        return self.net(x)

def score_matching_loss(model, x):
    batch_size = x.shape[0]
    x = x.requires_grad_(True)
    
    # Compute score
    score = model(x)
    
    # Term 2: 1/2 ||score||^2
    score_norm = 0.5 * (score ** 2).sum(dim=1)
    
    # Term 1: tr(nabla_x^2 log p_theta(x)) = tr(Jacobian of score)
    # We need to compute the trace of the Jacobian of score w.r.t. x
    trace = 0.0
    for i in range(x.shape[1]):  # For each dimension (x, y)
        # Compute gradient of score[i] w.r.t. x
        grad_outputs = torch.zeros_like(score)
        grad_outputs[:, i] = 1.0
        
        grad_score = torch.autograd.grad(
            outputs=score,
            inputs=x,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True
        )[0]
        
        # Add diagonal element to trace
        trace = trace + grad_score[:, i]
    
    loss = (trace + score_norm).mean()
    return loss


def dsm_loss(model, x, sigma=0.1):
    # 1. Perturb data with Gaussian noise
    noise = torch.randn_like(x) * sigma
    perturbed_x = x + noise
    
    # 2. Predict the score at the noisy point
    score = model(perturbed_x)
    
    # 3. Calculate the target score
    # For Gaussian noise N(0, sigma^2), the score is exactly (mean - x) / variance
    # Here: (x - perturbed_x) / sigma^2 = -noise / sigma^2
    target = -noise / (sigma ** 2)
    
    # 4. Minimize the squared error between predicted and target score
    loss = 0.5 * ((score - target) ** 2).sum(dim=1).mean()
    return loss