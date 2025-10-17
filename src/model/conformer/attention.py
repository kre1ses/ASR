import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

class RelativeMultiHeadSelfAttentionBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, attn_dropout_p: float = 0.1, 
                 layer_norm_eps: float = 1e-5, bias: bool = True):
        super().__init__()
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)

        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

        max_relative_position = 128
        self.max_relative_position = max_relative_position
        self.rel_pos_bias = nn.Parameter(
            torch.zeros((2 * max_relative_position - 1, num_heads))
        )

        self.input_ln = nn.LayerNorm(d_model, eps=layer_norm_eps)
        
        self.dropout = nn.Dropout(attn_dropout_p)
        self.attn_dropout = nn.Dropout(attn_dropout_p)
        
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            elif p.dim() == 1:
                nn.init.zeros_(p)
    
    def _relative_positional_bias(self, seq_len: int) -> torch.Tensor:
        device = self.rel_pos_bias.device

        range_vec = torch.arange(seq_len, device=device)
        relative_pos = range_vec[:, None] - range_vec[None, :]

        max_rel_pos = self.max_relative_position
        relative_pos.clamp_(-max_rel_pos + 1, max_rel_pos - 1)

        rel_pos_indices = relative_pos + max_rel_pos - 1

        rel_pos_bias_gathered = self.rel_pos_bias[rel_pos_indices]
        
        rel_pos_bias_gathered = rel_pos_bias_gathered.transpose(0, 2).transpose(1, 2)
        
        return rel_pos_bias_gathered
    
    def forward(self, 
                query: torch.Tensor, 
                key: torch.Tensor, 
                value: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                key_padding_mask: Optional[torch.Tensor] = None,
                attn_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        
        batch_size, seq_len, _ = query.shape
        
        query = self.input_ln(query)
        key = self.input_ln(key)
        value = self.input_ln(value)
        
        Q = self.q_proj(query).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.k_proj(key).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.v_proj(value).view(batch_size, seq_len, self.num_heads, self.d_v).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        rel_pos_bias = self._relative_positional_bias(seq_len)
        rel_pos_bias = rel_pos_bias.unsqueeze(0)
        
        scores = scores + rel_pos_bias.to(scores.device)
        
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask.unsqueeze(1), float('-inf'))
        
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(key_padding_mask, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )

        output = self.out_proj(attn_output)
        
        return output, attn_weights

class RelativeMultiHeadSelfAttentionModule(nn.Module):
    def __init__(self, d_model: int, num_heads: int, p_dropout: float = 0.1, 
                 layer_norm_eps: float = 1e-5, max_relative_position: int = 128):
        super().__init__()
        
        self.attention = RelativeMultiHeadSelfAttentionBlock(
            d_model, num_heads, attn_dropout_p=p_dropout, 
            layer_norm_eps=layer_norm_eps
        )
        self.layer_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout = nn.Dropout(p_dropout)
        self.max_relative_position = max_relative_position
    
    def forward(self, x: torch.Tensor, output_length: Optional[torch.Tensor] = None) -> torch.Tensor:

        batch_size, seq_len, _ = x.shape
        device = x.device
        
        attn_mask = None
        key_padding_mask = None
        
        if output_length is not None:
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), 
                diagonal=1
            )
            attn_mask = causal_mask.unsqueeze(0)
            
            seq_range = torch.arange(seq_len, device=device)
            key_padding_mask = seq_range[None, :] >= output_length[:, None].to(device)
        
        residual = x
        attn_output, attn_weights = self.attention(
            x, x, x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask
        )
        
        output = self.layer_norm(attn_output + residual)
        output = self.dropout(output)
        
        return output