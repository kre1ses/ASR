from math import floor

import torch
from torch import nn

from .bigru import LayerNormBiGRU

class DeepSpeechV2Model(nn.Module):
    def __init__(
            self, 
            input_dim, 
            n_tokens, 
            n_layers=3, 
            fc_hidden=512,
            n_channels=[32, 32], 
            kernel_size=[(11, 41), (11, 21)], 
            stride=[(2, 2), (1, 2)], 
            padding=[(5, 20), (5, 10)], 
            **batch):
        
        super().__init__()

        n_channels = [1] + n_channels

        self.n_channels = n_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        convs = []

        for i in range(len(kernel_size)):
            layer = nn.Sequential(
                nn.Conv2d(n_channels[i], n_channels[i + 1], kernel_size[i], stride[i], padding[i]),
                nn.Hardtanh(0, 20, inplace=True),
                nn.BatchNorm2d(n_channels[i+1])
            )
            convs.append(layer)
        
        self.convs = nn.Sequential(*convs)

        input_size = self.compute_shapes(input_dim, index=1) * n_channels[-1]

        self.rnn = LayerNormBiGRU(input_size=input_size, hidden_size=fc_hidden,
                                  num_layers=n_layers)

        self.linear = nn.Linear(2 * fc_hidden, n_tokens)

    def forward(
            self, 
            spectrogram: torch.Tensor, 
            spectrogram_lengths: torch.Tensor, 
            **batch
            ) -> tuple[torch.Tensor, torch.Tensor]:
        
        spectrogram = torch.unsqueeze(spectrogram, 1)
        spectrogram = spectrogram.transpose(2, 3)
        conv_out = self.convs(spectrogram.transpose(2, 3))
        conv_out = conv_out.view(conv_out.shape[0], conv_out.shape[2], -1)
        rnn_out = self.rnn(conv_out)
        output = self.linear(rnn_out)

        log_probs = nn.functional.softmax(output, dim=-1)

        log_probs_length = self.compute_shapes(spectrogram_lengths, index=0)

        d = {
            'log_probs': log_probs,
            'log_probs_length': log_probs_length
        }
        return d

    def compute_shapes(self, input_size: torch.Tensor, index: int) -> torch.Tensor:
        for i in range(len(self.kernel_size)):
            numerator = input_size + 2 * self.padding[i][index] - (self.kernel_size[i][index] - 1) - 1
            denominator = self.stride[i][index]

            if torch.is_tensor(input_size):
                input_size = torch.floor(numerator / denominator + 1).to(int)
            else:
                input_size = floor(numerator / denominator + 1)
                
        return input_size