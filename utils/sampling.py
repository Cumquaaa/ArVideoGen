import torch
from losses import DiffusionLoss

def sample(model, diffusion_model, device, sample_size=64):
    diffusion_model.eval()
    with torch.no_grad():
        noise = torch.randn(sample_size, 1, 28, 28).to(device)
        z = model(noise)
        samples = diffusion_model.sample(z, noise)
        samples = samples.view(sample_size, 1, 28, 28)
    return samples
