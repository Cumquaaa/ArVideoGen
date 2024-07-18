import torch
import torch.nn as nn

def diffusion_loss(x_recon, x):
    '''Dummy implementation of diffusion loss'''
    criterion = nn.MSELoss()
    loss = criterion(x_recon, x)  # No need to reshape x as both are now in [batch_size, 1, 28, 28]
    return loss

