import torch
import torch.nn as nn

class AutoregressiveModel(nn.Module):
    def __init__(self):
        super(AutoregressiveModel, self).__init__()
        self.fc = nn.Linear(28 * 28, 128)
        self.relu = nn.ReLU()
        self.fc_out = nn.Linear(128, 128)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.relu(x)
        x = self.fc_out(x)
        return x
