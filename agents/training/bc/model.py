from typing import List
import torch.nn as nn
import torch.nn.functional as F


class MLPBC(nn.Module):
    def __init__(self, input_dim: int, hidden: List[int] = [128, 128], dropout: float = 0.1):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        self.fe = nn.Sequential(*layers) if layers else nn.Identity()
        self.head = nn.Linear(prev, 1)

    def forward(self, x):
        z = self.fe(x)
        logits = self.head(z).squeeze(-1)
        return logits


class DeepMLPBC(nn.Module):
    """Deeper MLP with residual connections for better gradient flow."""
    def __init__(self, input_dim: int, hidden: List[int] = [512, 256, 256, 128], dropout: float = 0.0):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden[0])
        self.input_norm = nn.LayerNorm(hidden[0])
        
        self.blocks = nn.ModuleList()
        for i in range(len(hidden) - 1):
            self.blocks.append(nn.Sequential(
                nn.Linear(hidden[i], hidden[i + 1]),
                nn.LayerNorm(hidden[i + 1]),
                nn.GELU(),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity()
            ))
        
        self.head = nn.Linear(hidden[-1], 1)
    
    def forward(self, x):
        x = F.gelu(self.input_norm(self.input_proj(x)))
        for block in self.blocks:
            x = block(x)
        return self.head(x).squeeze(-1)
