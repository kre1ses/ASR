import torch
import torch.nn as nn

from .ffn import FeedForwardModule
from .attention import RelativeMultiHeadSelfAttentionModule
from .convolution import ConvModule
from .residual import ResidualConnectionModule


class ConformerBlock(nn.Module):

    def __init__(
            self,
            encoder_dim: int = 512,
            num_heads: int = 8,
            attn_dropout_p: float = 0.1,

            ffn_expansion_factor: int = 4,
            ffn_dropout_p: float = 0.1,

            conv_dropout_p: float = 0.1,
            conv_kernel_size: int = 10,
    ):
        super().__init__()

        self.ffn_1 = ResidualConnectionModule(
            module=FeedForwardModule(
                    encoder_dim=encoder_dim,
                    expansion_factor=ffn_expansion_factor,
                    dropout_p=ffn_dropout_p,
                ),
                module_factor = 0.5,
            )
        
        self.mhsa = ResidualConnectionModule(
                module=RelativeMultiHeadSelfAttentionModule(
                    d_model=encoder_dim,
                    num_heads=num_heads,
                    attn_dropout_p=attn_dropout_p,
                ),
                module_factor = 1.0
            )

        self.conv = ResidualConnectionModule(
                module=ConvModule(
                    in_channels=encoder_dim,
                    kernel_size=conv_kernel_size,
                    dropout_p=conv_dropout_p,
                ),
                module_factor = 1.0
            )
        
        self.ffn_2 = ResidualConnectionModule(
            module=FeedForwardModule(
                    encoder_dim=encoder_dim,
                    expansion_factor=ffn_expansion_factor,
                    dropout_p=ffn_dropout_p,
                ),
                module_factor = 0.5,
            )
        
        self.layer_norm = nn.LayerNorm(encoder_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ffn_1(x)
        x = self.mhsa(x)
        x = self.conv(x)
        x = self.ffn_2(x)
        x = self.layer_norm(x)
        return x