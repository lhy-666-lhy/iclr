"""Cross-arm hand feature fusion → single robot feature.

F_L, F_R [B,D] → F_robot [B,D]

fusion_type:
  - gated: F_L'=G(F_L,F_R), F_R'=G(F_R,F_L) (independent G), then Linear(2D→D)
  - mlp:   concat(F_L,F_R) → Linear→Act→…→Linear(D)
           optional mlp_hidden_dims overrides intermediate widths
           activation ∈ {gelu, relu}

Does not modify SPACE / encoders / training pipeline.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn
from torch import Tensor

_ACT: dict[str, type[nn.Module]] = {
    "gelu": nn.GELU,
    "relu": nn.ReLU,
}

class _GatedResidual(nn.Module):
    """G(A,B) = A + A ⊙ σ(MLP(Concat(A,B))), MLP hidden uses GELU."""

    def __init__(self, feature_dim: int, hidden_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, feature_dim),
        )

    def forward(self, a: Tensor, b: Tensor) -> Tensor:
        gate = torch.sigmoid(self.mlp(torch.cat([a, b], dim=-1)))
        return a + a * gate

class GatedHandFusion(nn.Module):
    """Bidirectional gated residual cross-arm fusion → project to [B,D]."""

    def __init__(self, feature_dim: int, hidden_dim: int):
        super().__init__()
        # Left/right G do not share parameters.
        self.g_left = _GatedResidual(feature_dim, hidden_dim)
        self.g_right = _GatedResidual(feature_dim, hidden_dim)
        self.proj = nn.Linear(2 * feature_dim, feature_dim)

    def forward(self, left_feature: Tensor, right_feature: Tensor) -> Tensor:
        f_l = self.g_left(left_feature, right_feature)   # G(F_L, F_R)
        f_r = self.g_right(right_feature, left_feature)  # G(F_R, F_L)
        return self.proj(torch.cat([f_l, f_r], dim=-1))  # [B, D]

class MLPHandFusion(nn.Module):
    """Concat MLP: Linear(2D → h1) → Act → … → Linear(hk → D).

    ``hidden_dims`` lists intermediate widths after the 2D input.
    Examples (D=128):
      [128]       → 256→128→128          (default ab1_ex2, GELU)
      [256]       → 256→256→128          (ab1ex2_hp1)
      [128, 128]  → 256→128→128→128      (ab1ex2_hp2)
      [64]        → 256→64→128           (ab1ex2_hp3)
      [128]+relu  → 256→128→128          (ab1ex2_hp4)
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 128,
        hidden_dims: Optional[Sequence[int]] = None,
        activation: str = "gelu",
    ):
        super().__init__()
        act_key = str(activation).lower()
        if act_key not in _ACT:
            raise ValueError(f"activation must be one of {tuple(_ACT)}, got {activation!r}")
        d = int(feature_dim)
        widths = [int(h) for h in hidden_dims] if hidden_dims is not None else [int(hidden_dim)]
        if not widths:
            raise ValueError("mlp hidden_dims must be non-empty")
        act_cls = _ACT[act_key]
        layers: list[nn.Module] = []
        in_dim = 2 * d
        for h in widths:
            layers.append(nn.Linear(in_dim, h))
            layers.append(act_cls())
            in_dim = h
        layers.append(nn.Linear(in_dim, d))
        self.mlp = nn.Sequential(*layers)
        self.hidden_dims = tuple(widths)
        self.activation = act_key

    def forward(self, left_feature: Tensor, right_feature: Tensor) -> Tensor:
        return self.mlp(torch.cat([left_feature, right_feature], dim=-1))

class HandFeatureFusion(nn.Module):
    """Dispatch cross-arm fusion by fusion_type ∈ {gated, mlp}."""

    VALID_TYPES = ("gated", "mlp")

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 2048,
        fusion_type: str = "gated",
        mlp_hidden_dims: Optional[Sequence[int]] = None,
        mlp_activation: str = "gelu",
        debug: bool = False,
    ):
        super().__init__()
        if fusion_type not in self.VALID_TYPES:
            raise ValueError(f"fusion_type must be one of {self.VALID_TYPES}, got {fusion_type}")
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.fusion_type = fusion_type
        self.mlp_hidden_dims = (
            tuple(int(h) for h in mlp_hidden_dims) if mlp_hidden_dims is not None else None
        )
        self.mlp_activation = str(mlp_activation).lower()
        self.debug = bool(debug)

        if fusion_type == "gated":
            self.fusion: nn.Module = GatedHandFusion(feature_dim, hidden_dim)
        else:
            self.fusion = MLPHandFusion(
                feature_dim,
                hidden_dim=hidden_dim,
                hidden_dims=self.mlp_hidden_dims,
                activation=self.mlp_activation,
            )

    def forward(self, left_feature: Tensor, right_feature: Tensor) -> Tensor:
        if left_feature.ndim != 2 or right_feature.ndim != 2:
            raise ValueError(
                f"expected [B,D] features, got left={tuple(left_feature.shape)} "
                f"right={tuple(right_feature.shape)}"
            )
        if left_feature.shape != right_feature.shape:
            raise ValueError(
                f"left/right shape mismatch: {tuple(left_feature.shape)} vs "
                f"{tuple(right_feature.shape)}"
            )
        if left_feature.shape[-1] != self.feature_dim:
            raise ValueError(
                f"feature_dim={self.feature_dim} but got D={left_feature.shape[-1]}"
            )

        robot_feature = self.fusion(left_feature, right_feature)

        if self.debug:
            extra = ""
            if self.fusion_type == "mlp":
                if self.mlp_hidden_dims is not None:
                    extra += f" mlp_hidden_dims={self.mlp_hidden_dims}"
                extra += f" act={self.mlp_activation}"
            print(
                f"[HandFeatureFusion] type={self.fusion_type}{extra} "
                f"left={tuple(left_feature.shape)} right={tuple(right_feature.shape)} "
                f"→ robot={tuple(robot_feature.shape)}"
            )
        return robot_feature

def _run_self_test() -> None:
    torch.manual_seed(0)
    B, D = 4, 128
    cases = {
        "gated": ("gated", None, "gelu"),
        "mlp_default": ("mlp", None, "gelu"),
        "mlp_hp1": ("mlp", [256], "gelu"),
        "mlp_hp2": ("mlp", [128, 128], "gelu"),
        "mlp_hp3": ("mlp", [64], "gelu"),
        "mlp_hp4": ("mlp", None, "relu"),  # 256→128→128 + ReLU
    }
    for name, (ftype, dims, act) in cases.items():
        fusion = HandFeatureFusion(
            feature_dim=D,
            hidden_dim=128,
            fusion_type=ftype,
            mlp_hidden_dims=dims,
            mlp_activation=act,
            debug=True,
        )
        fusion.train()
        left = torch.randn(B, D, requires_grad=True)
        right = torch.randn(B, D, requires_grad=True)
        out = fusion(left, right)
        assert out.shape == (B, D), (name, out.shape)
        out.pow(2).mean().backward()
        if ftype == "mlp":
            assert fusion.fusion.activation == act, (name, fusion.fusion.activation)
        print(f"  PASS {name} out={tuple(out.shape)}")

    print("=== hand_feature_fusion self-test PASS ===")

if __name__ == "__main__":
    _run_self_test()
