import math
import torch
import torch.nn as nn

import torch
import torch.nn as nn
import math

class RelativeSinusoidalPositionEmbedding(nn.Module):
    """
    Relative Sinusoidal Positional Encoding (RPE)
    Produces embeddings for relative positions (i - j)
    """
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
    
    def apply_rpe(self, tens: torch.Tensor) -> torch.Tensor:

        forward_pe = torch.zeros(tens.size(1), self.d_model)
        backward_pe = torch.zeros(tens.size(1), self.d_model)
        position = torch.arange(0, tens.size(1), dtype=torch.float32).unsqueeze(1)
        division = (10000**(torch.arange(0, self.d_model, 2, dtype=torch.float32) / self.d_model))

        forward_pe[:, 0::2] = torch.sin(position / division)
        forward_pe[:, 1::2] = torch.cos(position / division)
        backward_pe[:, 0::2] = torch.sin(-1 * position / division)
        backward_pe[:, 1::2] = torch.cos(-1 * position / division)

        forward_pe = torch.flip(forward_pe, [0]).unsqueeze(0)
        backward_pe = backward_pe[1:].unsqueeze(0)
        pe = torch.cat([forward_pe, backward_pe], dim=1)
        self.tens = pe.to(device=tens.device, dtype=tens.dtype)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.apply_rpe(x)
        
        encodings = self.tens[:, self.tens.size(1)//2 - x.size(1) + 1 : self.tens.size(1)//2 + x.size(1), :]
        return encodings
