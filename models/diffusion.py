import torch
import torch.nn as nn

class DiffusionNetwork(nn.Module):
    def __init__(self):
        super(DiffusionNetwork, self).__init__()
        self.fc = nn.Linear(128, 128)  # Adjust dimensions based on Autoregressive model output
        self.relu = nn.ReLU()
        self.fc_out = nn.Linear(128, 28 * 28)
    
    def forward(self, z):
        z = self.fc(z)
        z = self.relu(z)
        x_recon = self.fc_out(z)
        x_recon = x_recon.view(-1, 1, 28, 28)  # Reshape to the original image dimensions
        return x_recon
