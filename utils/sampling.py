import torch

def sample(model, device, sample_size=64):
    model.eval()
    with torch.no_grad():
        noise = torch.randn(sample_size, 28 * 28).to(device)
        samples = model(noise)
        samples = samples.view(sample_size, 1, 28, 28)
    return samples
