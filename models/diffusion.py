import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleMLP(nn.Module):
    def __init__(self, depth, width):
        super(SimpleMLP, self).__init__()
        self.depth = depth
        self.width = width
        layers = []
        for _ in range(depth):
            layers.append(nn.Linear(width, width))
            layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)
    
    def forward(self, x_t, timestep, z):
        batch_size = x_t.size(0)
        
        # Flatten x_t
        x_t_flat = x_t.view(batch_size, -1)
        
        # Expand timestep to match dimensions
        timestep_expanded = timestep.expand(batch_size, x_t_flat.size(1))
        
        # Concatenate along the last dimension
        combined_input = x_t_flat + timestep_expanded + z
        return self.net(combined_input)

class GaussianDiffusion:
    def __init__(self):
        self.num_timesteps = 1000  # Example number of timesteps
    
    def q_sample(self, x, timestep, noise):
        # Simplified example of q_sample
        return x + noise * (0.1 * timestep.view(-1, 1, 1, 1))
    
    def p_sample(self, net, x, t, z):
        # Simplified example of p_sample
        noise_pred = net(x, t, z)
        noise_pred = noise_pred.view(x.size())  # Reshape noise_pred to match x's shape
        return x - noise_pred * (0.1 * t.view(-1, 1, 1, 1))

class DiffusionLoss(nn.Module):
    def __init__(self, depth=2, width=1024):
        super(DiffusionLoss, self).__init__()
        self.net = SimpleMLP(depth, width)
        self.diffusion = GaussianDiffusion()
    
    def forward(self, z, x):
        noise = torch.randn_like(x)
        timestep = torch.randint(0, self.diffusion.num_timesteps, (x.size(0),), device=x.device).float()
        timestep = timestep.view(-1, 1)
        
        x_t = self.diffusion.q_sample(x, timestep, noise)
        
        # Flatten x_t and expand timestep for concatenation
        x_t_flat = x_t.view(x.size(0), -1)
        timestep_expanded = timestep.expand(x.size(0), x_t_flat.size(1))
        
        noise_pred = self.net(x_t, timestep_expanded, z)
        
        loss = F.mse_loss(noise_pred, noise.view(noise.size(0), -1))
        return loss
    
    def sample(self, z, noise):
        x = noise
        for t in reversed(range(self.diffusion.num_timesteps)):
            timestep = torch.tensor([t], device=x.device).float().view(-1, 1)
            timestep_expanded = timestep.expand(x.size(0), 1)
            x = self.diffusion.p_sample(self.net, x, timestep_expanded, z)
        return x

