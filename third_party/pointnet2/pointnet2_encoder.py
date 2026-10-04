"""Fast PointNet++ global encoder for robot PCs [B,N,C]→[B,out].

Scaled for N≈16–144. Tiny clouds (topology M≤32) use a single global SA
(equivalent to hierarchical PN++ degenerating gracefully) to avoid huge
B×K Conv2d activations.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from termcolor import cprint

from .pointnet2_utils import PointNetSetAbstraction

class PointNet2EncoderXYZ(nn.Module):
    CHUNK = 4096  # topology B×K can be >10k; chunk to limit peak VRAM

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 128,
        use_layernorm: bool = True,
        final_norm: str = "layernorm",
        use_projection: bool = True,
        **kwargs,
    ):
        super().__init__()
        del use_layernorm
        if in_channels < 3:
            raise ValueError(f"need >=3 channels, got {in_channels}")
        self.in_channels = int(in_channels)
        feat_dim = self.in_channels - 3

        # Geometry-scale path (N≈144): FPS(32)+kNN → global SA
        self.sa1 = PointNetSetAbstraction(
            npoint=32,
            radius=0.2,
            nsample=12,
            in_channel=3 + feat_dim,
            mlp=[64, 64, 128],
            group_all=False,
        )
        self.sa2 = PointNetSetAbstraction(
            npoint=None,
            radius=None,
            nsample=None,
            in_channel=128 + 3,
            mlp=[128, 128, 256],
            group_all=True,
        )
        # Topology-scale path (N≤32): PointNet-style MLP+maxpool
        # (group_all SA degenerates to this; Linear is much faster than Conv2d)
        self.tiny_mlp = nn.Sequential(
            nn.Linear(self.in_channels, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 256),
        )

        if final_norm == "layernorm":
            self.final_projection = nn.Sequential(
                nn.Linear(256, out_channels), nn.LayerNorm(out_channels)
            )
        elif final_norm == "none":
            self.final_projection = nn.Linear(256, out_channels)
        else:
            raise NotImplementedError(final_norm)
        if not use_projection:
            self.final_projection = nn.Identity()
        cprint(
            f"[PointNet2EncoderXYZ] FAST robot-scale in={in_channels} out={out_channels}",
            "cyan",
        )

    def _forward_chunk(self, x: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        x = x[..., : self.in_channels]
        if N <= 32:
            feat = self.tiny_mlp(x).max(dim=1)[0]
            return self.final_projection(feat)
        xyz = x[..., :3].transpose(1, 2).contiguous()
        points = x[..., 3:].transpose(1, 2).contiguous() if self.in_channels > 3 else None
        l1_xyz, l1_points = self.sa1(xyz, points)
        _, feat = self.sa2(l1_xyz, l1_points)
        return self.final_projection(feat.view(B, -1))

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