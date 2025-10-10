import torch
import torch.nn as nn

from .convolution import Conv2dSubsampling
from .block import ConformerBlock

class ConformerEncoder(nn.Module):

    def __init__(
            self,
            input_dim: int = 80,
            encoder_dim: int = 512,

            input_dropout_p: float = 0.1,

            num_layers: int = 17,
            num_heads: int = 8,
            attn_dropout_p: float = 0.1,

            ffn_expansion_factor: int = 4,
            ffn_dropout_p: float = 0.1,

            conv_dropout_p: float = 0.1,
            conv_kernel_size: int = 31,
    ):
        super().__init__() 
        self.conv_subsample = Conv2dSubsampling(
            out_dim = encoder_dim,
            in_dim = input_dim, 
            p_dropout = input_dropout_p
            )

        self.layers = nn.ModuleList([ConformerBlock(
            encoder_dim=encoder_dim,
            num_heads=num_heads,
            attn_dropout_p=attn_dropout_p,

            ffn_expansion_factor=ffn_expansion_factor,
            ffn_dropout_p=ffn_dropout_p,

            conv_dropout_p=conv_dropout_p,
            conv_kernel_size=conv_kernel_size
            ) for _ in range(num_layers)])

    def forward(self, inputs: torch.Tensor, input_lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        outputs, output_lengths = self.conv_subsample(inputs, input_lengths)

        for layer in self.layers:
            outputs = layer(outputs, output_lengths)

        return outputs, output_lengths