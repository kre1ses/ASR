import torch
import torch.nn as nn


class ResidualConnectionModule(nn.Module):

    def __init__(self, module: nn.Module, module_factor: float = 1.0):
        super(ResidualConnectionModule, self).__init__()
        self.module = module
        self.module_factor = module_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.module(x) * self.module_factor) + x