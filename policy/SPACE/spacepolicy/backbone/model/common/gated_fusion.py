"""Gated residual fusion: G(A,B) = A + A ⊙ σ(MLP([A;B])), MLP uses GELU."""

from __future__ import annotations

from typing import List, Optional, Sequence

import torch
import torch.nn as nn

class GatedFusion(nn.Module):
    """G(A, B) = A + A ⊙ Sigmoid(MLP(Concat(A, B))).

    - MLP hidden layers use GELU; last linear has no activation (pre-sigmoid).
    - Output of MLP has the same last-dim as A so the Hadamard product is valid.
    - All ops are differentiable (no detach); gradients flow to A, B and MLP params.
    """

    def __init__(
        self,
        dim_a: int,
        dim_b: Optional[int] = None,
        hidden_dims: Optional[Sequence[int]] = None,
    ):
        super().__init__()
        dim_b = dim_a if dim_b is None else int(dim_b)
        hidden: List[int] = list(hidden_dims) if hidden_dims is not None else [max(dim_a, 64)]

        layers: List[nn.Module] = []
        in_dim = dim_a + dim_b
        for h in hidden:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.GELU())
            in_dim = h
        layers.append(nn.Linear(in_dim, dim_a))  # logits → Sigmoid in forward
        self.mlp = nn.Sequential(*layers)
        self.dim_a = dim_a
        self.dim_b = dim_b

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Args:
            a: (..., dim_a)
            b: (..., dim_b)  — leading dims must broadcast / match a

        Returns:
            g: (..., dim_a)  same shape as a
        """
        if a.shape[-1] != self.dim_a:
            raise ValueError(f"A last dim {a.shape[-1]} != dim_a={self.dim_a}")
        if b.shape[-1] != self.dim_b:
            raise ValueError(f"B last dim {b.shape[-1]} != dim_b={self.dim_b}")
        if a.shape[:-1] != b.shape[:-1]:
            raise ValueError(f"A/B leading shape mismatch: {a.shape[:-1]} vs {b.shape[:-1]}")

        gate = torch.sigmoid(self.mlp(torch.cat([a, b], dim=-1)))  # (..., dim_a)
        return a + a * gate

__all__ = ["GatedFusion"]
