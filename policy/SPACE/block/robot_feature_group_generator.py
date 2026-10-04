"""Robot Feature Group Generator.

qpos [B,14] → left/right feature groups via make_robot + geometry/topology encoders.

Flags:
  use_geometry / use_topology — if False, do not construct or run that encoder
  (and skip the corresponding point-cloud path).

Feature group layout depends on flags:
  both:     [F_geometry, F_topology_1, ..., F_topology_K]
  geo only: [F_geometry]
  topo only: [F_topology_1, ..., F_topology_K]
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
from torch import Tensor

BLOCK_DIR = Path(__file__).resolve().parent
SPACE_ROOT = BLOCK_DIR.parent / "spacepolicy"
if str(BLOCK_DIR) not in sys.path:
    sys.path.insert(0, str(BLOCK_DIR))
if str(SPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(SPACE_ROOT))

from make_robot import RobotPointCloudGenerator  # noqa: E402
from backbone.model.vision.pointnet_extractor import (  # noqa: E402
    build_robot_geometry_encoder,
    build_robot_topology_encoder,
)

RobotTopologyEncoder = None  # set only if needed; builders cover both

class RobotFeatureGroupGenerator(nn.Module):
    """qpos → {left_feature_group, right_feature_group}."""

    def __init__(
        self,
        robot_generator: RobotPointCloudGenerator,
        left_geometry_encoder: Optional[nn.Module] = None,
        right_geometry_encoder: Optional[nn.Module] = None,
        left_topology_encoder: Optional[nn.Module] = None,
        right_topology_encoder: Optional[nn.Module] = None,
        use_geometry: bool = True,
        use_topology: bool = True,
        debug: bool = True,
    ):
        super().__init__()
        if not use_geometry and not use_topology:
            raise ValueError("need at least one of use_geometry / use_topology")

        self.robot_generator = robot_generator
        self.use_geometry = bool(use_geometry)
        self.use_topology = bool(use_topology)
        self.debug = bool(debug)

        # Optional modules: missing side is not registered → no params / no load keys
        if self.use_geometry:
            if left_geometry_encoder is None or right_geometry_encoder is None:
                raise ValueError("geometry encoders required when use_geometry=True")
            self.left_geometry_encoder = left_geometry_encoder
            self.right_geometry_encoder = right_geometry_encoder
        else:
            self.left_geometry_encoder = None
            self.right_geometry_encoder = None

        if self.use_topology:
            if left_topology_encoder is None or right_topology_encoder is None:
                raise ValueError("topology encoders required when use_topology=True")
            self.left_topology_encoder = left_topology_encoder
            self.right_topology_encoder = right_topology_encoder
        else:
            self.left_topology_encoder = None
            self.right_topology_encoder = None

    def _device(self) -> torch.device:
        try:
            return next(self.parameters()).device
        except StopIteration:
            return self.robot_generator.device

    def _apply(self, fn):
        out = super()._apply(fn)
        try:
            self.robot_generator.to(self._device())
        except Exception:
            pass
        return out

    @classmethod
    def from_defaults(
        cls,
        out_channels: int = 128,
        points_per_link: int = 48,
        use_layernorm: bool = True,
        final_norm: str = "layernorm",
        device: Union[str, torch.device] = "cpu",
        debug: bool = True,
        use_geometry: bool = True,
        use_topology: bool = True,
        pc_backbone: str = "pointnet",
        **generator_kwargs,
    ) -> "RobotFeatureGroupGenerator":
        """Build generator + only the encoders required by the flags.

        ``pc_backbone`` controls robot geometry + topology encoders only.
        Scene SPACE PC encoder is always official Simple PointNet.
        """
        if not use_geometry and not use_topology:
            raise ValueError("need at least one of use_geometry / use_topology")

        device = torch.device(device)
        robot_generator = RobotPointCloudGenerator(
            points_per_link=points_per_link, device=device, **generator_kwargs
        )

        left_geo = right_geo = left_topo = right_topo = None
        if use_geometry:
            geo_cfg = dict(
                backbone=pc_backbone,
                in_channels=3,
                out_channels=out_channels,
                use_layernorm=use_layernorm,
                final_norm=final_norm,
            )
            left_geo = build_robot_geometry_encoder(**geo_cfg)
            right_geo = build_robot_geometry_encoder(**geo_cfg)
        if use_topology:
            topo_cfg = dict(
                backbone=pc_backbone,
                in_channels=5,
                out_channels=out_channels,
                use_layernorm=use_layernorm,
                final_norm=final_norm,
            )
            left_topo = build_robot_topology_encoder(**topo_cfg)
            right_topo = build_robot_topology_encoder(**topo_cfg)

        return cls(
            robot_generator=robot_generator,
            left_geometry_encoder=left_geo,
            right_geometry_encoder=right_geo,
            left_topology_encoder=left_topo,
            right_topology_encoder=right_topo,
            use_geometry=use_geometry,
            use_topology=use_topology,
            debug=debug,
        ).to(device)

    def _encode_topology_group(
        self, topology_pc: Tensor, encoder: nn.Module
    ) -> List[Tensor]:
        """topology_pc: [B, K, M, 5] → list of K tensors [B, D]."""
        if topology_pc.ndim != 4 or topology_pc.shape[-1] != 5:
            raise ValueError(
                f"topology_pc expected [B,K,M,5], got {tuple(topology_pc.shape)}"
            )
        B, K, M, C = topology_pc.shape
        flat = topology_pc.reshape(B * K, M, C)
        feat_flat = encoder(flat)
        D = feat_flat.shape[-1]
        feat = feat_flat.reshape(B, K, D)
        return [feat[:, k, :] for k in range(K)]
    def forward(
        self,
        qpos: Tensor,
        robot_pc: Optional[Dict[str, Tensor]] = None,
    ) -> Dict[str, List[Tensor]]:
        """
        Returns left/right feature groups. Layout depends on use_geometry / use_topology.

        If ``robot_pc`` is provided (precomputed), skip online make_robot / FK.
        Expected keys (as needed by flags):
          left_geometry_pc, right_geometry_pc  [B,N,3]
          left_topology_pc, right_topology_pc  [B,K,M,5]
        """
        device = self._device()
        q = torch.as_tensor(qpos, dtype=torch.float32, device=device)
        if q.ndim == 1:
            q = q.unsqueeze(0)
        if q.ndim != 2 or q.shape[-1] != 14:
            raise ValueError(f"qpos expected [B,14], got {tuple(q.shape)}")

        B = q.shape[0]
        if self.debug:
            print(
                f"[RobotFeatureGroupGenerator] qpos={tuple(q.shape)} "
                f"use_geometry={self.use_geometry} use_topology={self.use_topology} "
                f"cached_pc={robot_pc is not None}"
            )

        # ---- Point clouds: cache or online FK ----
        left_geo = right_geo = left_topo = right_topo = None
        if robot_pc is not None:
            if self.use_geometry:
                left_geo = robot_pc["left_geometry_pc"]
                right_geo = robot_pc["right_geometry_pc"]
            if self.use_topology:
                left_topo = robot_pc["left_topology_pc"]
                right_topo = robot_pc["right_topology_pc"]
        elif self.use_geometry and self.use_topology:
            online = self.robot_generator.make_robot(q)
            left_geo = online["left_geometry_pc"]
            right_geo = online["right_geometry_pc"]
            left_topo = online["left_topology_pc"]
            right_topo = online["right_topology_pc"]
        elif self.use_geometry:
            geo = self.robot_generator.generate_geometry_pc(q)
            left_geo = geo["left_geometry_pc"]
            right_geo = geo["right_geometry_pc"]
        else:
            topo = self.robot_generator.generate_topology_pc(q)
            left_topo = topo["left_topology_pc"]
            right_topo = topo["right_topology_pc"]

        def _batchify(x: Optional[Tensor]) -> Optional[Tensor]:
            if x is None:
                return None
            return x.unsqueeze(0) if x.ndim == 2 else (x if x.ndim >= 3 else x)

        # geometry: [B,N,3]; topology after batchify: [B,K,M,5]
        if left_geo is not None and left_geo.ndim == 2:
            left_geo = left_geo.unsqueeze(0)
            right_geo = right_geo.unsqueeze(0)
        if left_topo is not None:
            if left_topo.ndim == 3:
                # (K,M,5) → (B,K,M,5)
                left_topo = left_topo.unsqueeze(0)
                right_topo = right_topo.unsqueeze(0)

        left_features: List[Tensor] = []
        right_features: List[Tensor] = []

        if self.use_geometry:
            assert self.left_geometry_encoder is not None
            f_geo_l = self.left_geometry_encoder(left_geo)
            f_geo_r = self.right_geometry_encoder(right_geo)
            left_features.append(f_geo_l)
            right_features.append(f_geo_r)
            if self.debug:
                print(f"[RobotFeatureGroupGenerator] F_geometry: {tuple(f_geo_l.shape)}")

        if self.use_topology:
            assert self.left_topology_encoder is not None
            f_topo_l = self._encode_topology_group(left_topo, self.left_topology_encoder)
            f_topo_r = self._encode_topology_group(right_topo, self.right_topology_encoder)
            left_features.extend(f_topo_l)
            right_features.extend(f_topo_r)
            if self.debug and f_topo_l:
                print(
                    f"[RobotFeatureGroupGenerator] F_topology K={len(f_topo_l)} "
                    f"dim={tuple(f_topo_l[0].shape)}"
                )

        if not left_features:
            raise RuntimeError("empty feature group")

        assert all(t.shape[0] == B for t in left_features + right_features)
        if self.debug:
            print(f"[RobotFeatureGroupGenerator] group_len={len(left_features)}")

        return {
            "left_feature_group": left_features,
            "right_feature_group": right_features,
        }

def _run_self_test() -> None:
    device = torch.device("cpu")
    for ug, ut, name in [
        (True, True, "both"),
        (True, False, "geo"),
        (False, True, "topo"),
    ]:
        mod = RobotFeatureGroupGenerator.from_defaults(
            out_channels=32,
            points_per_link=8,
            device=device,
            debug=False,
            use_geometry=ug,
            use_topology=ut,
        )
        # Missing side must not appear in named_parameters
        names = [n for n, _ in mod.named_parameters()]
        has_geo = any("geometry" in n for n in names)
        has_topo = any("topology" in n or "torpology" in n.lower() for n in names)
        # Class names are geometry_encoder / topology_encoder via attribute path
        has_geo = any("geometry_encoder" in n for n in names)
        has_topo = any("topology_encoder" in n for n in names)
        assert has_geo == ug, (name, names)
        assert has_topo == ut, (name, names)

        qpos = torch.randn(2, 14, device=device, requires_grad=True)
        out = mod(qpos)
        loss = sum(f.pow(2).mean() for f in out["left_feature_group"])
        loss.backward()
        assert qpos.grad is not None and qpos.grad.abs().sum() > 0
        print(f"  PASS {name}: group_len={len(out['left_feature_group'])} n_params={len(names)}")
    print("=== RobotFeatureGroupGenerator self-test PASS ===")

if __name__ == "__main__":
    _run_self_test()
