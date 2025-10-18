import torch
import torch.nn as nn

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


class FeedForwardModule(nn.Module):
    def __init__(
            self,
            encoder_dim: int = 512,
            expansion_factor: int = 4,
            dropout_p: float = 0.1,
    ) -> None:
        super().__init__()
        self.sequential = nn.Sequential(
            nn.LayerNorm(encoder_dim),
            nn.Linear(encoder_dim, encoder_dim * expansion_factor, bias=True),
            nn.SiLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(encoder_dim * expansion_factor, encoder_dim, bias=True),
            nn.Dropout(p=dropout_p),
        )
        register_gradient_hooks(self, name_prefix="FeedForwardModule")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sequential(x)