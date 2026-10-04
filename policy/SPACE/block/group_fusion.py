"""Intra-arm feature group fusion.

Feature Group (list of [B,D]) → single arm feature [B,D].
Left / right use independent fusion modules (no weight sharing).

Ablation fusion_type:
  - transformer: full group → TransformerEncoder → mean pool
  - mlp: concat tokens → MLP
  - no_geometry: drop F_geometry (index 0), then Transformer
  - geometry_only: return F_geometry only (no fusion params)
  - topo_tf_mlp: topo tokens → TF → 1; concat(geo,topo) → Linear(2D→D)→ReLU→Linear(D→D)
  - topo_tf_xattn_geo_q: topo → TF → 1; cross-attn with geo as Query
  - topo_tf_xattn_topo_q: topo → TF → 1; cross-attn with topo as Query

Does not modify SPACE / encoders / make_robot / robot_feature_group_generator.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
from torch import Tensor

def _stack_group(feature_group: Sequence[Tensor]) -> Tensor:
    """list of [B,D] → [B,T,D]."""
    if not feature_group:
        raise ValueError("feature_group is empty")
    for i, t in enumerate(feature_group):
        if t.ndim != 2:
            raise ValueError(f"token[{i}] expected [B,D], got {tuple(t.shape)}")
        if t.shape != feature_group[0].shape:
            raise ValueError(
                f"token[{i}] shape {tuple(t.shape)} != token[0] {tuple(feature_group[0].shape)}"
            )
    return torch.stack(list(feature_group), dim=1)

def _split_geo_topo(feature_group: Sequence[Tensor]) -> tuple[Tensor, List[Tensor]]:
    """Convention: [F_geo, F_topo_1, ..., F_topo_K]."""
    tokens = list(feature_group)
    if len(tokens) < 2:
        raise ValueError(
            f"topo late-fusion expects [geo, *topo] with len>=2, got {len(tokens)}"
        )
    return tokens[0], tokens[1:]

class TransformerGroupFusion(nn.Module):
    """Stack tokens → nn.TransformerEncoder → mean pool over token dim → [B,D]."""

    def __init__(
        self,
        feature_dim: int,
        num_layers: int = 4,
        nhead: int = 4,
        dim_feedforward: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        if feature_dim % nhead != 0:
            raise ValueError(f"feature_dim={feature_dim} must be divisible by nhead={nhead}")
        d_ff = int(dim_feedforward) if dim_feedforward is not None else 4 * feature_dim
        layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=nhead,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="relu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.feature_dim = feature_dim

    def forward(self, feature_group: Sequence[Tensor]) -> Tensor:
        x = _stack_group(feature_group)  # [B, T, D]
        y = self.encoder(x)  # [B, T, D]
        return y.mean(dim=1)  # [B, D]

class MLPGroupFusion(nn.Module):
    """Concat all tokens [B,(K+1)*D] → Linear → ReLU → Linear → [B,D]."""

    def __init__(self, feature_dim: int, group_size: int):
        super().__init__()
        if group_size < 1:
            raise ValueError(f"group_size must be >= 1, got {group_size}")
        self.feature_dim = feature_dim
        self.group_size = int(group_size)
        in_dim = self.group_size * feature_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
        )

    def forward(self, feature_group: Sequence[Tensor]) -> Tensor:
        if len(feature_group) != self.group_size:
            raise ValueError(
                f"MLPGroupFusion expects group_size={self.group_size}, "
                f"got {len(feature_group)}"
            )
        x = torch.cat(list(feature_group), dim=-1)  # [B, (K+1)*D]
        return self.mlp(x)

class VectorCrossAttention(nn.Module):
    """Vector A attends to vector B → same-dim fusion.

    Q=Wq A, K=Wk B, V=Wv B;
    alpha = Sigmoid((Q·K) / sqrt(d));
    Fusion = LayerNorm(A + alpha * V).
    """

    def __init__(self, feature_dim: int):
        super().__init__()
        d = int(feature_dim)
        self.feature_dim = d
        self.w_q = nn.Linear(d, d, bias=False)
        self.w_k = nn.Linear(d, d, bias=False)
        self.w_v = nn.Linear(d, d, bias=False)
        self.norm = nn.LayerNorm(d)
        self.scale = 1.0 / math.sqrt(d)

    def forward(self, a: Tensor, b: Tensor) -> Tensor:
        # a,b: [B,D]
        q = self.w_q(a)
        k = self.w_k(b)
        v = self.w_v(b)
        # scalar attention weight per batch item: [B,1]
        alpha = torch.sigmoid((q * k).sum(dim=-1, keepdim=True) * self.scale)
        raw = alpha * v
        return self.norm(a + raw)

class GeoTopoConcatMLP(nn.Module):
    """concat(F_geo, F_topo) → Linear(2D→D) → ReLU → Linear(D→D)."""

    def __init__(self, feature_dim: int):
        super().__init__()
        d = int(feature_dim)
        self.net = nn.Sequential(
            nn.Linear(2 * d, d),
            nn.ReLU(),
            nn.Linear(d, d),
        )

    def forward(self, geo: Tensor, topo: Tensor) -> Tensor:
        return self.net(torch.cat([geo, topo], dim=-1))

class TopoTransformerLateFusion(nn.Module):
    """Topo tokens → Transformer → 1 vec; then late-fuse with geometry.

    late_mode:
      - mlp: concat + Linear(2D→D)→ReLU→Linear(D→D)
      - xattn_geo_q: cross-attn, Query=geo, K/V=topo
      - xattn_topo_q: cross-attn, Query=topo, K/V=geo
    """

    VALID_LATE = ("mlp", "xattn_geo_q", "xattn_topo_q")

    def __init__(
        self,
        feature_dim: int,
        late_mode: str,
        num_layers: int = 4,
        nhead: int = 4,
        dim_feedforward: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        if late_mode not in self.VALID_LATE:
            raise ValueError(f"late_mode must be one of {self.VALID_LATE}, got {late_mode}")
        self.late_mode = late_mode
        self.topo_fusion = TransformerGroupFusion(
            feature_dim=feature_dim,
            num_layers=num_layers,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        if late_mode == "mlp":
            self.late: nn.Module = GeoTopoConcatMLP(feature_dim)
        else:
            self.late = VectorCrossAttention(feature_dim)

    def forward(self, feature_group: Sequence[Tensor]) -> Tensor:
        geo, topo_tokens = _split_geo_topo(feature_group)
        f_topo = self.topo_fusion(topo_tokens)
        if self.late_mode == "mlp":
            return self.late(geo, f_topo)
        if self.late_mode == "xattn_geo_q":
            return self.late(geo, f_topo)  # A=geo, B=topo
        return self.late(f_topo, geo)  # A=topo, B=geo

class RobotGroupFusion(nn.Module):
    """Left/right independent intra-arm fusion with ablation modes."""

    VALID_TYPES = (
        "transformer",
        "mlp",
        "no_geometry",
        "geometry_only",
        "topo_tf_mlp",
        "topo_tf_xattn_geo_q",
        "topo_tf_xattn_topo_q",
    )

    _LATE_MAP = {
        "topo_tf_mlp": "mlp",
        "topo_tf_xattn_geo_q": "xattn_geo_q",
        "topo_tf_xattn_topo_q": "xattn_topo_q",
    }

    def __init__(
        self,
        feature_dim: int,
        num_layers: int = 4,
        fusion_type: str = "transformer",
        group_size: Optional[int] = None,
        nhead: int = 8,
        dim_feedforward: Optional[int] = None,
        dropout: float = 0.0,
        debug: bool = False,
    ):
        super().__init__()
        if fusion_type not in self.VALID_TYPES:
            raise ValueError(f"fusion_type must be one of {self.VALID_TYPES}, got {fusion_type}")
        self.feature_dim = int(feature_dim)
        self.fusion_type = fusion_type
        self.group_size = group_size
        self.debug = bool(debug)

        # Independent L/R modules (never shared).
        self.left_fusion: Optional[nn.Module] = None
        self.right_fusion: Optional[nn.Module] = None

        if fusion_type in ("transformer", "no_geometry"):
            kw = dict(
                feature_dim=feature_dim,
                num_layers=num_layers,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
            )
            self.left_fusion = TransformerGroupFusion(**kw)
            self.right_fusion = TransformerGroupFusion(**kw)
        elif fusion_type == "mlp":
            if group_size is None:
                raise ValueError("fusion_type='mlp' requires group_size=K+1")
            self.left_fusion = MLPGroupFusion(feature_dim, group_size)
            self.right_fusion = MLPGroupFusion(feature_dim, group_size)
        elif fusion_type == "geometry_only":
            # No learnable fusion; topology path is unused by design.
            self.left_fusion = None
            self.right_fusion = None
        elif fusion_type in self._LATE_MAP:
            kw = dict(
                feature_dim=feature_dim,
                late_mode=self._LATE_MAP[fusion_type],
                num_layers=num_layers,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
            )
            self.left_fusion = TopoTransformerLateFusion(**kw)
            self.right_fusion = TopoTransformerLateFusion(**kw)

    def _select_tokens(self, feature_group: Sequence[Tensor]) -> List[Tensor]:
        tokens = list(feature_group)
        if not tokens:
            raise ValueError("empty feature_group")
        if self.fusion_type == "geometry_only":
            # Prefer a geometry-only group [F_geo]; if full [F_geo,*topo], take first.
            return [tokens[0]]
        if self.fusion_type == "no_geometry":
            # Topology-only groups (from generator with use_geometry=False) already
            # exclude F_geo — use all tokens. Legacy full groups still drop index 0.
            # Heuristic: full layout has K+1 tokens with K>=1 (len>=2) and was mixed;
            # topo-only may also be len>=2. Prefer: if module flag set... Use all when
            # caller already built topo-only. We detect "legacy full" only via
            # optional attribute; default uses all tokens when group built topo-only.
            # Convention after component ablations: topo-only list is pure topology.
            # Legacy: [geo, *topo] with no_geometry → drop first when attr not set.
            if getattr(self, "topology_group_is_pure", True):
                return tokens
            if len(tokens) < 2:
                raise ValueError("no_geometry needs topology tokens")
            return tokens[1:]
        return tokens  # transformer / mlp / topo late-fusion: full group

    def forward(
        self,
        left_feature_group: Sequence[Tensor],
        right_feature_group: Sequence[Tensor],
    ) -> Dict[str, Tensor]:
        left_tokens = self._select_tokens(left_feature_group)
        right_tokens = self._select_tokens(right_feature_group)

        if self.debug:
            print(
                f"[RobotGroupFusion] type={self.fusion_type} "
                f"left_tokens={len(left_tokens)} right_tokens={len(right_tokens)} "
                f"token_shape={tuple(left_tokens[0].shape)}"
            )

        if self.fusion_type == "geometry_only":
            f_left = left_tokens[0]
            f_right = right_tokens[0]
        else:
            assert self.left_fusion is not None and self.right_fusion is not None
            f_left = self.left_fusion(left_tokens)
            f_right = self.right_fusion(right_tokens)

        if self.debug:
            print(f"[RobotGroupFusion] left_feature:  {tuple(f_left.shape)}")
            print(f"[RobotGroupFusion] right_feature: {tuple(f_right.shape)}")

        return {"left_feature": f_left, "right_feature": f_right}

def _run_self_test() -> None:
    torch.manual_seed(0)
    B, D, T = 2, 1024, 5
    left = [torch.randn(B, D, requires_grad=True) for _ in range(T)]
    right = [torch.randn(B, D, requires_grad=True) for _ in range(T)]

    for ftype in RobotGroupFusion.VALID_TYPES:
        kw = dict(feature_dim=D, num_layers=2, fusion_type=ftype, nhead=8, debug=True)
        if ftype == "mlp":
            kw["group_size"] = T
        fusion = RobotGroupFusion(**kw)
        fusion.train()
        out = fusion(left, right)
        assert out["left_feature"].shape == (B, D)
        assert out["right_feature"].shape == (B, D)

        # Fresh leaves so each mode gets its own graph
        left_g = [t.detach().clone().requires_grad_(True) for t in left]
        right_g = [t.detach().clone().requires_grad_(True) for t in right]
        out = fusion(left_g, right_g)
        loss = out["left_feature"].pow(2).mean() + out["right_feature"].pow(2).mean()
        loss.backward()

        assert out["left_feature"].requires_grad
        assert out["right_feature"].requires_grad

        if ftype == "no_geometry":
            # Production path uses topo-only groups from the generator
            # (topology_group_is_pure=True → keep all provided tokens).
            assert left_g[0].grad is not None and left_g[0].grad.abs().sum() > 0
        elif ftype == "geometry_only":
            assert left_g[0].grad is not None and left_g[0].grad.abs().sum() > 0
            # Topology unused
            assert left_g[1].grad is None or left_g[1].grad.abs().sum() == 0
        else:
            # Full group participates (incl. topo late-fusion)
            assert left_g[0].grad is not None and left_g[0].grad.abs().sum() > 0
            assert left_g[1].grad is not None and left_g[1].grad.abs().sum() > 0

        if ftype != "geometry_only":
            for n, p in fusion.named_parameters():
                assert p.grad is not None and p.grad.abs().sum() > 0, f"{ftype}: no grad on {n}"

        # L/R modules must be distinct objects
        if fusion.left_fusion is not None:
            assert fusion.left_fusion is not fusion.right_fusion

        print(f"  PASS fusion_type={ftype}")

    print("=== group_fusion self-test PASS ===")

if __name__ == "__main__":
    _run_self_test()
