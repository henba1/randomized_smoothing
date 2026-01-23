from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn


@dataclass(frozen=True)
class LoRAConfig:
    r: int = 8
    alpha: float = 16.0
    dropout: float = 0.0
    target_substrings: tuple[str, ...] = ("qkv", "proj", "fc1", "fc2")


class LoRALinear(nn.Module):
    """
    Minimal LoRA wrapper for nn.Linear.

    Forward: y = x W^T + b + (alpha/r) * ( (dropout(x) A^T) B^T )
    """

    def __init__(self, base: nn.Linear, *, r: int, alpha: float, dropout: float):
        super().__init__()
        if r <= 0:
            raise ValueError("LoRA rank r must be > 0")

        self.base = base
        self.r = int(r)
        self.alpha = float(alpha)
        self.scaling = float(alpha) / float(r)
        self.dropout = nn.Dropout(p=float(dropout)) if dropout > 0 else nn.Identity()

        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)

        dev = base.weight.device
        dt = base.weight.dtype
        self.lora_A = nn.Parameter(torch.zeros((self.r, base.in_features), device=dev, dtype=dt))
        self.lora_B = nn.Parameter(torch.zeros((base.out_features, self.r), device=dev, dtype=dt))

        # Common init: A ~ N(0, 0.01), B = 0 so the wrapper is initially a no-op.
        nn.init.normal_(self.lora_A, std=0.01)
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        x_d = self.dropout(x)
        # (B, *, in) @ (in, r) -> (B, *, r) ; then @(r, out) -> (B, *, out)
        lora = (x_d @ self.lora_A.t()) @ self.lora_B.t()
        return out + self.scaling * lora


def inject_lora_(model: nn.Module, cfg: LoRAConfig) -> int:
    """
    In-place inject LoRA into matching nn.Linear submodules.

    Returns:
        number of modules replaced
    """

    replaced = 0

    def _recurse(parent: nn.Module, prefix: str) -> None:
        nonlocal replaced
        for name, child in list(parent.named_children()):
            qname = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.Linear) and any(s in qname for s in cfg.target_substrings):
                setattr(parent, name, LoRALinear(child, r=cfg.r, alpha=cfg.alpha, dropout=cfg.dropout))
                replaced += 1
            else:
                _recurse(child, qname)

    _recurse(model, "")
    return replaced


def lora_parameters(model: nn.Module) -> list[nn.Parameter]:
    return [p for n, p in model.named_parameters() if "lora_A" in n or "lora_B" in n]


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v for k, v in model.state_dict().items() if ".lora_A" in k or ".lora_B" in k}


def save_lora_checkpoint(path: str, *, cfg: LoRAConfig, model: nn.Module, extra: dict[str, Any] | None = None) -> None:
    payload: dict[str, Any] = {
        "lora": asdict(cfg),
        "state_dict": lora_state_dict(model),
    }
    if extra:
        payload["extra"] = extra
    torch.save(payload, path)


def load_lora_checkpoint(path: str, device: torch.device | None = None) -> tuple[LoRAConfig, dict[str, torch.Tensor]]:
    ckpt = torch.load(path, map_location=device)
    cfg = LoRAConfig(**ckpt["lora"])
    state = ckpt["state_dict"]
    return cfg, state

