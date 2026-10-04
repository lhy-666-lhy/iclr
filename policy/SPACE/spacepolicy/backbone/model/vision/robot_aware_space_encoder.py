"""SPACE-compatible obs encoder: SPACEEncoder + RobotAwareEncoder.

Same interface as SPACEEncoder:
  forward(observations: Dict) -> [B, 192]
  output_shape() -> 192

Downstream diffusion / SPACE.policy wiring stays unchanged; only swap obs_encoder.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional, Union

import torch
import torch.nn as nn
from termcolor import cprint

from backbone.model.vision.pointnet_extractor import SPACEEncoder

# block/ sits next to spacepolicy under "SPACE copy"
_BLOCK_DIR = Path(__file__).resolve().parents[4] / "block"
if str(_BLOCK_DIR) not in sys.path:
    sys.path.insert(0, str(_BLOCK_DIR))

from robot_aware_encoder import SPACE_DIM, RobotAwareEncoder  # noqa: E402

class RobotAwareSPACEEncoder(nn.Module):
    """Upstream drop-in replacement for SPACEEncoder (output dim unchanged).

    Important:
      - SPACEEncoder.agent_pos uses *normalized* state (same as vanilla SPACE).
      - Robot FK / make_robot must use *raw* joint qpos in radians.
      Pass raw angles via observations['agent_pos_raw'] (injected by SPACE policy).
    """

    RAW_QPOS_KEY = "agent_pos_raw"
    ROBOT_PC_KEYS = (
        "left_geometry_pc",
        "right_geometry_pc",
        "left_topology_pc",
        "right_topology_pc",
    )

    def __init__(
        self,
        observation_space: Dict,
        img_crop_shape=None,
        out_channel=128,
        state_mlp_size=(64, 64),
        state_mlp_activation_fn=nn.ReLU,
        pointcloud_encoder_cfg=None,
        use_pc_color=False,
        pointnet_type="pointnet",
        # robot-aware
        robot_aware_encoder_type: str = "main",
        robot_aware_points_per_link: int = 16,
        robot_aware_group_num_layers: int = 4,
        robot_aware_hand_hidden_dim: int = 128,
        robot_aware_injection_hidden_dim: Optional[int] = None,
        robot_dim: Optional[int] = None,
        device: Union[str, torch.device] = "cpu",
        debug: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.state_key = "agent_pos"
        self.space_encoder = SPACEEncoder(
            observation_space=observation_space,
            img_crop_shape=img_crop_shape,
            out_channel=out_channel,
            state_mlp_size=state_mlp_size,
            state_mlp_activation_fn=state_mlp_activation_fn,
            pointcloud_encoder_cfg=pointcloud_encoder_cfg,
            use_pc_color=use_pc_color,
            pointnet_type=pointnet_type,
        )
        space_dim = self.space_encoder.output_shape()
        if space_dim != SPACE_DIM:
            raise ValueError(
                f"RobotAwareEncoder expects SPACE feature dim {SPACE_DIM} "
                f"(obs128+state64), got {space_dim}. "
                f"Keep encoder_output_dim=128 and state_mlp_size=(..., 64)."
            )

        rdim = int(robot_dim) if robot_dim is not None else int(out_channel)
        self.robot_aware = RobotAwareEncoder(
            encoder_type=robot_aware_encoder_type,
            robot_dim=rdim,
            points_per_link=robot_aware_points_per_link,
            group_num_layers=robot_aware_group_num_layers,
            hand_hidden_dim=robot_aware_hand_hidden_dim,
            injection_hidden_dim=robot_aware_injection_hidden_dim,
            device=device,
            debug=debug,
        )
        self.n_output_channels = space_dim
        self.debug = bool(debug)
        samp = getattr(self.robot_aware, "sample_region", "surface")
        m_pts = getattr(self.robot_aware, "points_per_link", robot_aware_points_per_link)
        robot_pc = getattr(self.robot_aware, "pc_backbone", "pointnet")
        inj_h = getattr(self.robot_aware, "injection_hidden_dim", None)
        cprint(
            f"[RobotAwareSPACEEncoder] type={robot_aware_encoder_type} "
            f"robot_pc={robot_pc} (geo+topo only) scene_pc=simple_pointnet "
            f"sample={samp} M={m_pts} hand_h={robot_aware_hand_hidden_dim} "
            f"inj_h={inj_h} out_dim={self.n_output_channels}; "
            f"FK uses '{self.RAW_QPOS_KEY}' (raw qpos) or precomputed robot_pc, "
            f"SPACE state uses normalized agent_pos",
            "green",
        )

    def _extract_robot_pc(self, observations: Dict) -> Optional[Dict[str, torch.Tensor]]:
        if not all(k in observations for k in self.ROBOT_PC_KEYS):
            # geo-only / topo-only ablations may omit unused keys; build partial dict
            present = {k: observations[k] for k in self.ROBOT_PC_KEYS if k in observations}
            return present if present else None
        return {k: observations[k] for k in self.ROBOT_PC_KEYS}

    def forward(self, observations: Dict) -> torch.Tensor:
        # Same contract as SPACEEncoder: batch may be B or B*To flattened.
        # Normalized agent_pos goes into SPACE state MLP (vanilla behavior).
        space_feat = self.space_encoder(observations)  # [B, 192]

        # Raw joint angles for make_robot FK (must NOT be LinearNormalizer output).
        if self.RAW_QPOS_KEY in observations:
            qpos = observations[self.RAW_QPOS_KEY]
        else:
            # Fallback for unit tests without policy wrapper — not for training.
            qpos = observations[self.state_key]
            cprint(
                f"[RobotAwareSPACEEncoder] WARNING: missing '{self.RAW_QPOS_KEY}', "
                f"falling back to '{self.state_key}' (may be normalized).",
                "red",
            )
        if qpos.ndim != 2 or qpos.shape[-1] != 14:
            raise ValueError(f"raw qpos expected [B,14], got {tuple(qpos.shape)}")
        if self.debug:
            print(
                f"[RobotAwareSPACEEncoder] raw_qpos range "
                f"[{float(qpos.min()):.3f}, {float(qpos.max()):.3f}]"
            )
        robot_pc = self._extract_robot_pc(observations)
        if self.debug:
            print(f"[RobotAwareSPACEEncoder] using_cached_robot_pc={robot_pc is not None}")
        return self.robot_aware(qpos, space_feat, robot_pc=robot_pc)  # [B, 192]

    def output_shape(self) -> int:
        return self.n_output_channels
