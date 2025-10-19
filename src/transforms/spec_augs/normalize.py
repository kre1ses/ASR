import torch
import torch.nn as nn

class Normalize1D(nn.Module):

    def __init__(self, mean: float = None, std: float = None, eps: float = 1e-8):
        super().__init__()
        self.mean = mean
        self.std = std
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = (x - self.mean) / (self.std + self.eps)

        return x.squeeze(1)