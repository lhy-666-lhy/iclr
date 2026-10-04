"""Factory for robot geometry / topology PC backbones only.

Scene DP3 encoder is always official Simple PointNet (not controlled here).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import torch.nn as nn

_TP = Path(__file__).resolve().parent
if str(_TP) not in sys.path:
    sys.path.insert(0, str(_TP))

VALID_PC_BACKBONES = ("pointnet", "pointnet2", "pointtransformer")

def normalize_backbone(name: str) -> str:
    bb = str(name).lower().strip()
    if bb in ("pn", "simple", "simple_pointnet"):
        return "pointnet"
    if bb in ("pn2", "pointnet++", "pointnet_pp"):
        return "pointnet2"
    if bb in ("pt", "point_transformer", "transformer"):
        return "pointtransformer"
    if bb not in VALID_PC_BACKBONES:
        raise ValueError(f"unknown robot pc backbone {name!r}, expected {VALID_PC_BACKBONES}")
    return bb

def build_robot_pc_encoder(
    backbone: str,
    in_channels: int,
    out_channels: int = 128,
    use_layernorm: bool = True,
    final_norm: str = "layernorm",
    use_projection: bool = True,
    pointnet_cls: Optional[type] = None,
    **kwargs,
) -> nn.Module:
    """Build encoder for robot geo (C=3) or topology (C=5) point clouds."""
    bb = normalize_backbone(backbone)
    common = dict(
        in_channels=in_channels,
        out_channels=out_channels,
        use_layernorm=use_layernorm,
        final_norm=final_norm,
        use_projection=use_projection,
    )
    if bb == "pointnet":
        if pointnet_cls is None:
            raise ValueError("pointnet_cls required for simple PointNet backbone")
        return pointnet_cls(**common, **kwargs)
    if bb == "pointnet2":
        from pointnet2 import PointNet2EncoderXYZ

        return PointNet2EncoderXYZ(**common, **kwargs)
    from point_transformer import PointTransformerEncoderXYZ

    return PointTransformerEncoderXYZ(**common, **kwargs)
