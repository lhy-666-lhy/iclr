"""Robot-aware encoder: qpos + SPACE features → 192-d conditioning.

Canonical experiment ids:  main | ab{N}_ex{M} | ab_exp{K}
  ab1_ex1 … ab1_ex4   Ablation-1 fusion
  ab2_ex2 … ab2_ex4   Ablation-2 components (no ab2_ex1 empty)
  ab3_ex1 … ab3_ex6   Ablation-3 sampling
  ab4_ex1 … ab4_ex3   Ablation-4 robot geo+topo PC backbone
  ab_exp1 … ab_exp6   Ablation-exp: topo-first late geo×topo (+ optional hand MLP)

Legacy short aliases still accepted (ex1, geo, surf8, pn2, …).
Scene SPACE PC encoder is always official Simple PointNet.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

BLOCK_DIR = Path(__file__).resolve().parent
SPACE_ROOT = BLOCK_DIR.parent / "spacepolicy"
if str(BLOCK_DIR) not in sys.path:
    sys.path.insert(0, str(BLOCK_DIR))
if str(SPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(SPACE_ROOT))

from group_fusion import RobotGroupFusion  # noqa: E402
from hand_feature_fusion import HandFeatureFusion  # noqa: E402
from robot_feature_group_generator import RobotFeatureGroupGenerator  # noqa: E402
from backbone.model.common.gated_fusion import GatedFusion  # noqa: E402

OBS_DIM = 128
STATE_DIM = 64
SPACE_DIM = OBS_DIM + STATE_DIM  # 192
ROBOT_DIM = 128  # rf

# encoder_type →
#   (hand, injection, group_fusion, use_geometry, use_topology,
#    sample_region, points_per_link_or_None, robot_pc_backbone)
_EncoderSpec = Tuple[str, str, str, bool, bool, str, Optional[int], str]

_ENCODER_SPECS: dict[str, _EncoderSpec] = {
    # ----- main (≈ ab1_ex4 = ab2_ex4 = ab3 surface M=16 recipe = ab4_ex1 backbone) -----
    "main": ("gated", "gated", "transformer", True, True, "surface", None, "pointnet"),

    # ----- Ablation-1 · Cross-Arm × Injection -----
    "ab1_ex1": ("mlp", "concat_mlp", "transformer", True, True, "surface", None, "pointnet"),
    "ab1_ex2": ("mlp", "gated", "transformer", True, True, "surface", None, "pointnet"),
    "ab1_ex3": ("gated", "concat_mlp", "transformer", True, True, "surface", None, "pointnet"),
    "ab1_ex4": ("gated", "gated", "transformer", True, True, "surface", None, "pointnet"),
    # ab1_ex2 hand-MLP width variants (same fusion recipe; only MLP widths differ)
    "ab1ex2_hp1": ("mlp", "gated", "transformer", True, True, "surface", None, "pointnet"),
    "ab1ex2_hp2": ("mlp", "gated", "transformer", True, True, "surface", None, "pointnet"),
    "ab1ex2_hp3": ("mlp", "gated", "transformer", True, True, "surface", None, "pointnet"),
    "ab1ex2_hp4": ("mlp", "gated", "transformer", True, True, "surface", None, "pointnet"),

    # ----- Ablation-2 · Geometry × Topology (EX1 empty not implemented) -----
    "ab2_ex2": ("gated", "gated", "geometry_only", True, False, "surface", None, "pointnet"),
    "ab2_ex3": ("gated", "gated", "transformer", False, True, "surface", None, "pointnet"),
    "ab2_ex4": ("gated", "gated", "transformer", True, True, "surface", None, "pointnet"),

    # ----- Ablation-3 · Sample region × M -----
    "ab3_ex1": ("gated", "gated", "transformer", True, True, "surface", 8, "pointnet"),
    "ab3_ex2": ("gated", "gated", "transformer", True, True, "surface", 16, "pointnet"),
    "ab3_ex3": ("gated", "gated", "transformer", True, True, "surface", 32, "pointnet"),
    "ab3_ex4": ("gated", "gated", "transformer", True, True, "inner", 8, "pointnet"),
    "ab3_ex5": ("gated", "gated", "transformer", True, True, "inner", 16, "pointnet"),
    "ab3_ex6": ("gated", "gated", "transformer", True, True, "inner", 32, "pointnet"),

    # ----- Ablation-4 · Robot geo+topo PC backbone only -----
    "ab4_ex1": ("gated", "gated", "transformer", True, True, "surface", None, "pointnet"),
    "ab4_ex2": ("gated", "gated", "transformer", True, True, "surface", None, "pointnet2"),
    "ab4_ex3": ("gated", "gated", "transformer", True, True, "surface", None, "pointtransformer"),

    # ----- Ablation-exp · topo-first late fusion (injection always gated) -----
    # exp1: topo→TF→1; concat(geo,topo)+MLP; hand=gated
    # exp2: topo→TF→1; cross-attn geo=Q; hand=gated
    # exp3: topo→TF→1; cross-attn topo=Q; hand=gated
    # exp4/5/6: same arm-internal as 1/2/3 but hand=mlp (ab1_ex2-style)
    "ab_exp1": ("gated", "gated", "topo_tf_mlp", True, True, "surface", None, "pointnet"),
    "ab_exp2": ("gated", "gated", "topo_tf_xattn_geo_q", True, True, "surface", None, "pointnet"),
    "ab_exp3": ("gated", "gated", "topo_tf_xattn_topo_q", True, True, "surface", None, "pointnet"),
    "ab_exp4": ("mlp", "gated", "topo_tf_mlp", True, True, "surface", None, "pointnet"),
    "ab_exp5": ("mlp", "gated", "topo_tf_xattn_geo_q", True, True, "surface", None, "pointnet"),
    "ab_exp6": ("mlp", "gated", "topo_tf_xattn_topo_q", True, True, "surface", None, "pointnet"),
}

_ALIASES: dict[str, str] = {
    "ex1": "ab1_ex1",
    "ex2": "ab1_ex2",
    "ex3": "ab1_ex3",
    "ex4": "ab1_ex4",
    "geo": "ab2_ex2",
    "topo": "ab2_ex3",
    "surf8": "ab3_ex1",
    "surf16": "ab3_ex2",
    "surf32": "ab3_ex3",
    "inner8": "ab3_ex4",
    "inner16": "ab3_ex5",
    "inner32": "ab3_ex6",
    "pn": "ab4_ex1",
    "pn2": "ab4_ex2",
    "pt": "ab4_ex3",
}
for _alias, _canon in _ALIASES.items():
    _ENCODER_SPECS[_alias] = _ENCODER_SPECS[_canon]

# Hand MLP intermediate widths for mlp fusion (D=128 → input 256).
# None → default single hidden = hand_hidden_dim (256→128→128 for ab1_ex2).
_MLP_HAND_HIDDEN_DIMS: dict[str, Optional[Tuple[int, ...]]] = {
    "ab1ex2_hp1": (256,),       # 256→256→128
    "ab1ex2_hp2": (128, 128),   # 256→128→128→128
    "ab1ex2_hp3": (64,),        # 256→64→128
    # ab1ex2_hp4: same widths as ab1_ex2 default; only activation differs
}

# Hand MLP nonlinearity; default gelu. ab1ex2_hp4 = ReLU on default 256→128→128.
_MLP_HAND_ACTIVATION: dict[str, str] = {
    "ab1ex2_hp4": "relu",
}

CANONICAL_TYPES = (
    "main",
    "ab1_ex1",
    "ab1_ex2",
    "ab1_ex3",
    "ab1_ex4",
    "ab1ex2_hp1",
    "ab1ex2_hp2",
    "ab1ex2_hp3",
    "ab1ex2_hp4",
    "ab2_ex2",
    "ab2_ex3",
    "ab2_ex4",
    "ab3_ex1",
    "ab3_ex2",
    "ab3_ex3",
    "ab3_ex4",
    "ab3_ex5",
    "ab3_ex6",
    "ab4_ex1",
    "ab4_ex2",
    "ab4_ex3",
    "ab_exp1",
    "ab_exp2",
    "ab_exp3",
    "ab_exp4",
    "ab_exp5",
    "ab_exp6",
)

def normalize_encoder_type(encoder_type: str) -> str:
    if encoder_type not in _ENCODER_SPECS:
        raise ValueError(
            f"unknown encoder_type={encoder_type!r}. "
            f"Use: {CANONICAL_TYPES} (or short aliases {tuple(_ALIASES.keys())})"
        )
    return encoder_type

def resolve_robot_pc_backbone(encoder_type: str) -> str:
    return _ENCODER_SPECS[normalize_encoder_type(encoder_type)][7]

def resolve_robot_pc_sampling(
    encoder_type: str, default_points_per_link: int = 16
) -> tuple[str, int]:
    """Return (sample_region, M) used by make_robot for this encoder_type."""
    encoder_type = normalize_encoder_type(encoder_type)
    sample_region = _ENCODER_SPECS[encoder_type][5]
    pts_override = _ENCODER_SPECS[encoder_type][6]
    m = int(pts_override) if pts_override is not None else int(default_points_per_link)
    return sample_region, m

def robot_pc_zarr_key_prefix(sample_region: str, points_per_link: int) -> str:
    return f"{sample_region}_m{int(points_per_link)}"

def robot_pc_zarr_keys(sample_region: str, points_per_link: int) -> dict[str, str]:
    """Map logical PC names → flat zarr data keys."""
    tag = robot_pc_zarr_key_prefix(sample_region, points_per_link)
    return {
        "left_geometry_pc": f"robot_geo_left_{tag}",
        "right_geometry_pc": f"robot_geo_right_{tag}",
        "left_topology_pc": f"robot_topo_left_{tag}",
        "right_topology_pc": f"robot_topo_right_{tag}",
    }

# All sampling variants needed by main + ablations (skip duplicates of main).
ROBOT_PC_PRECOMPUTE_VARIANTS: tuple[tuple[str, int], ...] = (
    ("surface", 8),
    ("surface", 16),
    ("surface", 32),
    ("inner", 8),
    ("inner", 16),
    ("inner", 32),
)

class ConcatMLPFusion(nn.Module):
    def __init__(self, dim_a: int, dim_b: int, out_dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        h = int(hidden_dim) if hidden_dim is not None else max(dim_a + dim_b, out_dim)
        self.net = nn.Sequential(
            nn.Linear(dim_a + dim_b, h),
            nn.GELU(),
            nn.Linear(h, out_dim),
        )

    def forward(self, a: Tensor, b: Tensor) -> Tensor:
        return self.net(torch.cat([a, b], dim=-1))

def _split_space(space_feature: Tensor) -> tuple[Tensor, Tensor]:
    if space_feature.ndim != 2 or space_feature.shape[-1] != SPACE_DIM:
        raise ValueError(f"space_feature expected [B,{SPACE_DIM}], got {tuple(space_feature.shape)}")
    return space_feature[:, :OBS_DIM], space_feature[:, OBS_DIM:]

def _build_robot_stack(
    hand_fusion_type: str,
    group_fusion_type: str = "transformer",
    use_geometry: bool = True,
    use_topology: bool = True,
    sample_region: str = "surface",
    pc_backbone: str = "pointnet",
    robot_dim: int = ROBOT_DIM,
    points_per_link: int = 16,
    group_num_layers: int = 4,
    hand_hidden_dim: int = 128,
    mlp_hidden_dims: Optional[Sequence[int]] = None,
    mlp_activation: str = "gelu",
    device: Union[str, torch.device] = "cpu",
    debug_group_gen: bool = False,
):
    device = torch.device(device)
    gen = RobotFeatureGroupGenerator.from_defaults(
        out_channels=robot_dim,
        points_per_link=points_per_link,
        device=device,
        debug=debug_group_gen,
        use_geometry=use_geometry,
        use_topology=use_topology,
        sample_region=sample_region,
        pc_backbone=pc_backbone,
    )
    nhead = 8 if robot_dim % 8 == 0 else 4
    if robot_dim % nhead != 0:
        nhead = 1
    gfusion = RobotGroupFusion(
        feature_dim=robot_dim,
        num_layers=group_num_layers,
        fusion_type=group_fusion_type,
        group_size=None,
        nhead=nhead,
        debug=False,
    )
    hfusion = HandFeatureFusion(
        feature_dim=robot_dim,
        hidden_dim=hand_hidden_dim,
        fusion_type=hand_fusion_type,
        mlp_hidden_dims=mlp_hidden_dims,
        mlp_activation=mlp_activation,
        debug=False,
    )
    return gen, gfusion, hfusion

class FusionRobotAwareEncoder(nn.Module):
    def __init__(
        self,
        robot_group_generator: RobotFeatureGroupGenerator,
        group_fusion: RobotGroupFusion,
        hand_fusion: HandFeatureFusion,
        obs_fusion: nn.Module,
        state_fusion: nn.Module,
        debug: bool = False,
    ):
        super().__init__()
        self.robot_group_generator = robot_group_generator
        self.group_fusion = group_fusion
        self.hand_fusion = hand_fusion
        self.obs_fusion = obs_fusion
        self.state_fusion = state_fusion
        self.debug = bool(debug)

    def compute_robot_feature(
        self, qpos: Tensor, robot_pc: Optional[Dict[str, Tensor]] = None
    ) -> Tensor:
        groups = self.robot_group_generator(qpos, robot_pc=robot_pc)
        arms = self.group_fusion(groups["left_feature_group"], groups["right_feature_group"])
        rf = self.hand_fusion(arms["left_feature"], arms["right_feature"])
        if self.debug:
            print(f"[FusionRobotAwareEncoder] rf: {tuple(rf.shape)}")
        return rf

    def fuse_space(self, space_feature: Tensor, rf: Tensor) -> Tensor:
        obs_f, state_f = _split_space(space_feature)
        out = torch.cat([self.obs_fusion(obs_f, rf), self.state_fusion(state_f, rf)], dim=-1)
        if self.debug:
            print(f"[FusionRobotAwareEncoder] out: {tuple(out.shape)}")
        return out

    def forward(
        self,
        qpos: Tensor,
        space_input: Tensor,
        robot_pc: Optional[Dict[str, Tensor]] = None,
    ) -> Tensor:
        return self.fuse_space(space_input, self.compute_robot_feature(qpos, robot_pc=robot_pc))

def _make_injection(
    injection_type: str,
    robot_dim: int,
    gated_hidden: Sequence[int],
    injection_hidden_dim: Optional[int] = None,
) -> tuple[nn.Module, nn.Module]:
    """Build obs/state injection modules.

    - gated: GatedFusion MLP widths = gated_hidden
    - concat_mlp: ConcatMLPFusion hidden = injection_hidden_dim
      (default max(dim_a+dim_b, out_dim) when None)
    """
    if injection_type == "gated":
        return (
            GatedFusion(dim_a=OBS_DIM, dim_b=robot_dim, hidden_dims=list(gated_hidden)),
            GatedFusion(dim_a=STATE_DIM, dim_b=robot_dim, hidden_dims=list(gated_hidden)),
        )
    if injection_type == "concat_mlp":
        return (
            ConcatMLPFusion(OBS_DIM, robot_dim, OBS_DIM, hidden_dim=injection_hidden_dim),
            ConcatMLPFusion(STATE_DIM, robot_dim, STATE_DIM, hidden_dim=injection_hidden_dim),
        )
    raise ValueError(f"unknown injection_type: {injection_type}")

class RobotAwareEncoder(nn.Module):
    """encoder_type: main | abN_exM (canonical) or short alias."""

    VALID_TYPES = tuple(_ENCODER_SPECS.keys())
    CANONICAL_TYPES = CANONICAL_TYPES

    def __init__(
        self,
        encoder_type: str = "main",
        robot_dim: int = ROBOT_DIM,
        points_per_link: int = 16,
        group_num_layers: int = 4,
        hand_hidden_dim: int = 128,
        injection_hidden_dim: Optional[int] = None,
        gated_hidden: Optional[Sequence[int]] = None,
        device: Union[str, torch.device] = "cpu",
        debug: bool = False,
    ):
        super().__init__()
        encoder_type = normalize_encoder_type(encoder_type)
        self.encoder_type = encoder_type
        self.robot_dim = robot_dim
        self.debug = bool(debug)
        device = torch.device(device)
        # Injection MLP width: explicit gated_hidden > injection_hidden_dim > default
        if gated_hidden is not None:
            gh = list(gated_hidden)
        elif injection_hidden_dim is not None:
            gh = [int(injection_hidden_dim)]
        else:
            gh = [max(robot_dim, 128)]

        (
            hand_type,
            inj_type,
            group_type,
            use_geo,
            use_topo,
            sample_region,
            pts_override,
            pc_backbone,
        ) = _ENCODER_SPECS[encoder_type]
        m = int(pts_override) if pts_override is not None else int(points_per_link)
        self.sample_region = sample_region
        self.points_per_link = m
        self.pc_backbone = pc_backbone
        self.hand_hidden_dim = int(hand_hidden_dim)
        self.injection_hidden_dim = (
            int(injection_hidden_dim) if injection_hidden_dim is not None else None
        )

        mlp_h = _MLP_HAND_HIDDEN_DIMS.get(encoder_type)
        mlp_act = _MLP_HAND_ACTIVATION.get(encoder_type, "gelu")
        gen, gfusion, hfusion = _build_robot_stack(
            hand_fusion_type=hand_type,
            group_fusion_type=group_type,
            use_geometry=use_geo,
            use_topology=use_topo,
            sample_region=sample_region,
            pc_backbone=pc_backbone,
            robot_dim=robot_dim,
            points_per_link=m,
            group_num_layers=group_num_layers,
            hand_hidden_dim=hand_hidden_dim,
            mlp_hidden_dims=mlp_h,
            mlp_activation=mlp_act,
            device=device,
        )
        obs_fusion, state_fusion = _make_injection(
            inj_type, robot_dim, gh, injection_hidden_dim=injection_hidden_dim
        )
        self.encoder: nn.Module = FusionRobotAwareEncoder(
            robot_group_generator=gen,
            group_fusion=gfusion,
            hand_fusion=hfusion,
            obs_fusion=obs_fusion,
            state_fusion=state_fusion,
            debug=debug,
        )
        self.to(device)

    def forward(
        self,
        qpos: Tensor,
        space_input: Tensor,
        robot_pc: Optional[Dict[str, Tensor]] = None,
    ) -> Tensor:
        return self.encoder(qpos, space_input, robot_pc=robot_pc)

    def output_shape(self) -> int:
        return SPACE_DIM

MainRobotAwareEncoder = FusionRobotAwareEncoder

def _run_self_test() -> None:
    torch.manual_seed(0)
    B = 2
    device = torch.device("cpu")
    qpos = torch.randn(B, 14, device=device)
    space = torch.randn(B, SPACE_DIM, device=device)
    print("=== abN_exM smoke ===")
    for et in (
        "main",
        "ab1_ex1",
        "ab1_ex2",
        "ab1ex2_hp1",
        "ab1ex2_hp2",
        "ab1ex2_hp3",
        "ab1ex2_hp4",
        "ab2_ex2",
        "ab3_ex1",
        "ab4_ex2",
        "ab_exp1",
        "ab_exp2",
        "ab_exp3",
        "ab_exp4",
        "ab_exp5",
        "ab_exp6",
    ):
        enc = RobotAwareEncoder(
            encoder_type=et,
            points_per_link=8,
            group_num_layers=1,
            hand_hidden_dim=128,
            injection_hidden_dim=256,
            device=device,
        )
        out = enc(qpos, space)
        assert out.shape == (B, SPACE_DIM), (et, out.shape)
        mh = getattr(enc.encoder.hand_fusion, "mlp_hidden_dims", None)
        ma = getattr(enc.encoder.hand_fusion, "mlp_activation", None)
        print(f"  PASS {et} hand_h=128 inj_h=256 mlp_h={mh} act={ma}")
    print("=== PASS ===")

if __name__ == "__main__":
    _run_self_test()
