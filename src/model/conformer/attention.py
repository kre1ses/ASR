import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from .rpe import RelativeSinusoidalPositionEmbedding

def register_gradient_hooks(module: nn.Module, name_prefix: str = ""):
    """
    Регистрирует backward-хуки для всех параметров модуля,
    чтобы логировать норму, NaN и Inf в градиентах.
    """
    for name, param in module.named_parameters():
        if param.requires_grad:
            full_name = f"{name_prefix}.{name}" if name_prefix else name

            def hook(grad, pname=full_name):
                if grad is None:
                    print(f"[grad_logger] {pname}: grad=None")
                    return
                grad_norm = grad.norm().item()
                has_nan = torch.isnan(grad).any().item()
                has_inf = torch.isinf(grad).any().item()
                print(
                    f"[grad_logger] {pname:60s} | norm={grad_norm:10.3e} "
                    f"| NaN={has_nan} | Inf={has_inf}"
                )

            param.register_hook(hook)

class RelativeMultiHeadSelfAttentionBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, attn_dropout_p: float = 0.1):
        super().__init__()
        "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.Q = nn.Linear(d_model, d_model, bias=True)
        init.xavier_uniform_(self.Q.weight)
        init.zeros_(self.Q.bias)

        self.K = nn.Linear(d_model, d_model, bias=True)
        init.xavier_uniform_(self.K.weight)
        init.zeros_(self.K.bias)

        self.V= nn.Linear(d_model, d_model, bias=True)
        init.xavier_uniform_(self.V.weight)
        init.zeros_(self.V.bias)
        
        self.rpe = RelativeSinusoidalPositionEmbedding(d_model)
        self.rpe_proj = nn.Linear(d_model, d_model, bias=False)

        self.u_bias = nn.Parameter(torch.Tensor(self.num_heads, self.d_k))
        self.v_bias = nn.Parameter(torch.Tensor(self.num_heads, self.d_k))
        torch.nn.init.xavier_uniform_(self.u_bias)
        torch.nn.init.xavier_uniform_(self.v_bias)

        self.out_proj = nn.Linear(d_model, d_model, bias=True)
        init.xavier_uniform_(self.out_proj .weight)
        init.zeros_(self.out_proj .bias)
        
        self.dropout = nn.Dropout(attn_dropout_p)
    
    def forward(self, 
                query: torch.Tensor, 
                key: torch.Tensor, 
                value: torch.Tensor,
                pos_embedding: torch.Tensor, 
                mask: torch.Tensor = None) -> torch.Tensor:
        
        batch_size = query.size(0)

        Q = self.Q(query).view(batch_size, -1, self.num_heads, self.d_k)
        K = self.K(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.V(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        R = self.rpe_proj(pos_embedding).view(batch_size, -1, self.num_heads, self.d_k)

        main_score = torch.matmul((Q + self.u_bias).transpose(1, 2), K.transpose(2, 3))
        rpe_score = torch.matmul((Q + self.v_bias).transpose(1, 2), R.transpose(2, 3).transpose(1, 3)) # 0 1 2 3 -> 0 1 3 2 -> 0 2 3 1

        # Shift for correct dims (oh hell nah i hate ts)
        B, H, T1, T2 = rpe_score.size()
        zeros = rpe_score.new_zeros(B, H, T1, 1)
        padded_rpe_score = torch.cat([zeros, rpe_score], dim=-1)

        padded_rpe_score = padded_rpe_score.view(B, H, T2 + 1, T1)
        rpe_score = padded_rpe_score[:, :, 1:].view_as(rpe_score)[:, :, :, : T2 // 2 + 1]

        scores = (main_score + rpe_score) / math.sqrt(self.d_k)

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask, 1e-4) 

        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)

        output = torch.matmul(scores, V)
        output = output.contiguous().view(batch_size, -1, self.d_model)

        return self.out_proj(output)

class RelativeMultiHeadSelfAttentionModule(nn.Module):
    def __init__(self, d_model: int, num_heads: int, p_dropout: float = 0.1):
        super().__init__()

        self.attention = RelativeMultiHeadSelfAttentionBlock(d_model, num_heads, p_dropout)
        self.rpe = RelativeSinusoidalPositionEmbedding(d_model)
        self.dropout = nn.Dropout(p = p_dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        # register_gradient_hooks(self, name_prefix="AttentionModule")
    
    def forward(self, x: torch.Tensor, output_lenght: torch.Tensor = None) -> torch.Tensor:
        B, T, D = x.size()

        device = x.device
        seq_range = torch.arange(T, device=device)[None, :]  # (1, T)

        output_lenght = output_lenght.to(device)
        mask = seq_range >= output_lenght[:, None]
        # mask = mask.unsqueeze(1)  # (B, 1, T)

        pos_embedding = self.rpe(x)
        pos_embedding = pos_embedding.repeat(B, 1, 1)

        x = self.layer_norm(x)
        outputs = self.attention(x, x, x, pos_embedding=pos_embedding, mask=mask)

        return self.dropout(outputs)