import torch
from torch import nn


def classifier(kind, widths=(64, 64, 32)):
    if kind == 'linear':
        return nn.Linear(4, 2)
    if kind != 'mlp':
        raise ValueError(f'Unknown model: {kind}')
    layers = []
    previous = 4
    for i, width in enumerate(widths):
        layers.extend([nn.Linear(previous, width), nn.GELU()])
        if i == 0:
            layers.append(nn.LayerNorm(width))
        previous = width
    return nn.Sequential(*layers, nn.Linear(previous, 2))


class WeightHead(nn.Module):
    """Separate positive conditional-mean regressor for each class."""
    def __init__(self, scales):
        super().__init__()
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Linear(4, 32), nn.GELU(), nn.Linear(32, 32),
                          nn.GELU(), nn.Linear(32, 1)) for _ in range(2)
        ])
        self.register_buffer('scales', torch.as_tensor(scales, dtype=torch.float32))
        for head in self.heads:
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)

    def forward(self, x):
        # Bounds are numerical safeguards; saturation is recorded in diagnostics.
        log_shape = torch.cat([head(x) for head in self.heads], dim=1).clamp(-15, 15)
        return log_shape + self.scales.log()
