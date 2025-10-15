import torch
from torch import nn


class LayerNormBiGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()

        self.first_GRU = nn.GRU(input_size=input_size, hidden_size=hidden_size,
                                batch_first=True, bidirectional=True, num_layers=1)
        other_GRU = []
        layer_norms = []
        for i in range(num_layers - 1):
            other_GRU.append(nn.GRU(input_size=2 * hidden_size, hidden_size=hidden_size,
                                    batch_first=True, bidirectional=True, num_layers=1))
            layer_norms.append(nn.LayerNorm(2 * hidden_size))
        self.other_GRU = nn.ModuleList(other_GRU)
        self.layer_norms = nn.ModuleList(layer_norms)

    def forward(self, input):
        output, h_n = self.first_GRU(input)
        for i in range(len(self.other_GRU)):
            output = self.layer_norms[i](output)
            output, h_n = self.other_GRU[i](output, h_n)
        return output    