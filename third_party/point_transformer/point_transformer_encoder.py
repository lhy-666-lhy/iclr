"""Fast Point Transformer encoder for robot PCs [B,N,C]→[B,out].

True local kNN attention + FPS downsample for large N.
Large batches (topology B×K) are automatically chunked to cap activation memory.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from termcolor import cprint

_TP = Path(__file__).resolve().parents[1]
if str(_TP) not in sys.path:
    sys.path.insert(0, str(_TP))
from pointnet2.pointnet2_utils import (  # noqa: E402
    farthest_point_sample,
    index_points,
    knn_point,
)

class PointTransformerBlock(nn.Module):
    """Local kNN attention with scalar weights: O(B·N·k·C)."""

    def __init__(self, d_model: int, k: int = 8):
        super().__init__()
        self.k = int(k)
        self.to_q = nn.Linear(d_model, d_model, bias=False)
        self.to_k = nn.Linear(d_model, d_model, bias=False)
        self.to_v = nn.Linear(d_model, d_model, bias=False)
        self.pos_mlp = nn.Sequential(
            nn.Linear(3, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
        )
        self.attn_mlp = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(inplace=True),
            nn.Linear(d_model // 2, 1),
        )
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, xyz: torch.Tensor, feats: torch.Tensor) -> torch.Tensor:
        N = xyz.shape[1]
        k = min(self.k, N)
        idx = knn_point(k, xyz, xyz)
        neigh_xyz = index_points(xyz, idx)
        rel = neigh_xyz - xyz.unsqueeze(2)
        pos = self.pos_mlp(rel)

        q = self.to_q(feats).unsqueeze(2)
        neigh_f = index_points(feats, idx)
        k_f = self.to_k(neigh_f)
        v_f = self.to_v(neigh_f)

        attn = self.attn_mlp(q - k_f + pos).squeeze(-1)
        attn = F.softmax(attn, dim=-1).unsqueeze(-1)
        agg = (attn * (v_f + pos)).sum(dim=2)
        feats = self.norm1(feats + agg)
        feats = self.norm2(feats + self.ff(feats))
        return feats

class PointTransformerEncoderXYZ(nn.Module):
    CHUNK = 2048  # cap activation memory for topology B×K

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 128,
        use_layernorm: bool = True,
        final_norm: str = "layernorm",
        use_projection: bool = True,
        d_model: int = 64,
        nhead: int = 4,
        num_attn_layers: int = 2,
        knn_k: int = 8,
        max_points: int = 48,
        dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        del use_layernorm, nhead, dropout
        if in_channels < 3:
            raise ValueError(f"need >=3 channels, got {in_channels}")
        self.in_channels = int(in_channels)
        self.d_model = int(d_model)
        self.max_points = int(max_points)

        self.input_proj = nn.Sequential(
            nn.Linear(self.in_channels, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList(
            [PointTransformerBlock(d_model, k=knn_k) for _ in range(int(num_attn_layers))]
        )

        if final_norm == "layernorm":
            self.final_projection = nn.Sequential(
                nn.Linear(d_model, out_channels), nn.LayerNorm(out_channels)
            )
        elif final_norm == "none":
            self.final_projection = nn.Linear(d_model, out_channels)
        else:
            raise NotImplementedError(final_norm)
        if not use_projection:
            self.final_projection = nn.Identity()
        cprint(
            f"[PointTransformerEncoderXYZ] FAST local-kNN "
            f"in={in_channels} out={out_channels} d_model={d_model} "
            f"k={knn_k} L={num_attn_layers} max_pts={max_points}",
            "cyan",
        )

    def _forward_chunk(self, x: torch.Tensor) -> torch.Tensor:
        x = x[..., : self.in_channels]
        xyz = x[..., :3]
        N = xyz.shape[1]

        # Tiny clouds (topology M≈16, huge B×K): MLP+pool only.
        # Local kNN attn on 10k×16×k tensors dominates VRAM; hierarchy is for N≳48.
        if N <= 32:
            feats = self.input_proj(x)
            return self.final_projection(feats.max(dim=1)[0])

        feats = self.input_proj(x)
        if N > self.max_points:
            fps_idx = farthest_point_sample(xyz, self.max_points)
            xyz = index_points(xyz, fps_idx)
            feats = index_points(feats, fps_idx)

        for blk in self.blocks:
            feats = blk(xyz, feats)
        return self.final_projection(feats.max(dim=1)[0])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"expected [B,N,C], got {tuple(x.shape)}")
        if x.shape[-1] < self.in_channels:
            raise ValueError(f"need C>={self.in_channels}, got {x.shape[-1]}")
        B = x.shape[0]
        if B <= self.CHUNK:
            return self._forward_chunk(x)
        outs = [self._forward_chunk(x[i : i + self.CHUNK]) for i in range(0, B, self.CHUNK)]
        return torch.cat(outs, dim=0)
