import torch
import torch.nn as nn
import torch.nn.init as init
import math

class DepthWiseConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()

        self.conv = nn.Conv1d(
            in_channels = in_channels, 
            out_channels = out_channels, 
            kernel_size = kernel_size, 
            groups = in_channels, 
            padding = (kernel_size - 1) // 2
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x) # (batch_size, embed_dim, seq_len)

class PointWiseConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()

        self.conv = nn.Conv1d(
            in_channels = in_channels,
            out_channels = out_channels,
            kernel_size = kernel_size,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x) # (batch_size, embed_dim, seq_len)

class ConvModule(nn.Module):
    def __init__(self, in_channels: int, kernel_size: int = 3, dropout_p: float = 0.1):
        super().__init__()

        self.layer_norm = nn.LayerNorm(in_channels) # (batch_size, seq_len, embed_dim)

        self.conv_module = nn.Sequential(
            PointWiseConv(in_channels, in_channels * 2, 1), # (batch_size, 2*embed_dim, seq_len)
            nn.GLU(dim=1),
            DepthWiseConv(in_channels, in_channels, kernel_size),
            nn.BatchNorm1d(in_channels),
            nn.SiLU(),
            PointWiseConv(in_channels, in_channels, 1),
            nn.Dropout(p=dropout_p),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # print(x.shape)
        x = self.layer_norm(x).transpose(1, 2)
        # x = self.layer_norm(x)
        # print(x.shape)
        x = self.conv_module(x)
        # print(x.shape)
        return x.transpose(1, 2)

class Conv2dSubsampling(nn.Module):
    def __init__(self, out_dim: int, in_dim: int, p_dropout: float = 0.1):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels = 1,
                out_channels = out_dim,
                kernel_size = 3,
                stride = 2,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels = out_dim,
                out_channels = out_dim,
                kernel_size = 3,
                stride = 2,
            ),
            nn.ReLU(),
        )

        self.linear = nn.Linear(out_dim * ((in_dim - 1) // 2 - 1) // 2, out_dim, bias=True)
        init.xavier_uniform_(self.linear.weight)
        init.zeros_(self.linear.bias)

        self.dropout = nn.Dropout(p = p_dropout)
    
    def forward(self, x: torch.Tensor, input_lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x.unsqueeze(1)  # (batch_size, 1, seq_len, freq)
        x = x.transpose(2, 3)  # (batch_size, 1, freq, seq_len)
        x = self.conv(x)  # (batch_size, channels, seq_len', freq')
        batch_size, channels, subsampled_seq_len, subsampled_freq = x.size()
        x = x.transpose(1, 2).contiguous().view(batch_size, subsampled_seq_len, channels * subsampled_freq)
        x = self.linear(x)
        x = self.dropout(x)

        # output_lengths = input_lengths >> 2
        # output_lengths -= 1
        output_lengths = torch.floor((input_lengths - 3) / 2 + 1)
        output_lengths = torch.floor((output_lengths - 3) / 2 + 1)

        return x, output_lengths