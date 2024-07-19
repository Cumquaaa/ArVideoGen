import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerBlock(nn.Module):
    def __init__(self, width):
        super(TransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(width)
        self.attn = nn.MultiheadAttention(embed_dim=width, num_heads=8)
        self.norm2 = nn.LayerNorm(width)
        self.fc1 = nn.Linear(width, 4 * width)
        self.fc2 = nn.Linear(4 * width, width)
    
    def forward(self, x):
        # Multi-Head Attention
        x_norm = self.norm1(x)
        attn_output, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_output
        
        # Feed Forward Network
        x_norm = self.norm2(x)
        x_ffn = self.fc2(F.relu(self.fc1(x_norm)))
        x = x + x_ffn
        return x

class AutoregressiveModel(nn.Module):
    def __init__(self, width=1024, depth=32):
        super(AutoregressiveModel, self).__init__()
        self.fc_in = nn.Linear(28 * 28, width)
        self.blocks = nn.ModuleList([TransformerBlock(width) for _ in range(depth)])
        self.norm = nn.LayerNorm(width)
        self.fc_out = nn.Linear(width, 784)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc_in(x).unsqueeze(0)  # Transformer expects [seq_len, batch_size, embed_dim]
        
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        x = x.squeeze(0)  # Remove the seq_len dimension
        x = self.fc_out(x)
        return x
