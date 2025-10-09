import torch
import torch.nn as nn
import torch.nn.init as init

from .encoder import ConformerEncoder


class Conformer(nn.Module):

    def __init__(
            self,
            vocab_size: int,
            input_dim: int = 80,
            encoder_dim: int = 512,
            input_dropout_p: float = 0.1,

            num_layers: int = 17,
            num_attn_heads: int = 8,
            attn_dropout_p: float = 0.1,

            feed_forward_expansion_factor: int = 4,
            feed_forward_dropout_p: float = 0.1,

            conv_dropout_p: float = 0.1,
            conv_kernel_size: int = 10
    ) -> None:
        super(Conformer, self).__init__()

        self.encoder = ConformerEncoder(
            input_dim=input_dim,
            encoder_dim=encoder_dim,
            input_dropout_p=input_dropout_p,

            num_layers=num_layers,
            num_attn_heads=num_attn_heads,
            attn_dropout_p=attn_dropout_p,

            feed_forward_expansion_factor=feed_forward_expansion_factor,
            feed_forward_dropout_p=feed_forward_dropout_p,

            conv_dropout_p=conv_dropout_p,
            conv_kernel_size=conv_kernel_size,
        )
        self.linear = nn.Linear(encoder_dim, vocab_size, bias=False)
        init.xavier_uniform_(self.linear.weight)

    def forward(
            self, 
            spectrogram: torch.Tensor, 
            spectrogram_lengths: torch.Tensor
            ) -> tuple[torch.Tensor, torch.Tensor]:

        encoder_outputs, log_probs_length = self.encoder(spectrogram, spectrogram_lengths)
        output = self.linear(encoder_outputs)
        log_probs = nn.functional.log_softmax(output, dim=-1)
        return log_probs, log_probs_length