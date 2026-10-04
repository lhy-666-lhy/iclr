"""Robot state → geometry/topology point clouds → feature tokens (no fusion).

Used only by PPI_rp1. Default embodiment is DualPanda (RLBench2).
Does NOT run RobotWin group/hand/injection fusion, and does NOT use AgileX
make_robot unless robot_embodiment is explicitly set to agilex/robotwin.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

ROBOT_PC_KEYS = (
    "left_geometry_pc",
    "right_geometry_pc",
    "left_topology_pc",
    "right_topology_pc",
)

def _repo_root() -> Path:
    # space/model/vision/*.py → parents[5] = repo root
    return Path(__file__).resolve().parents[5]

def _ensure_dp3_paths() -> None:
    root = _repo_root()
    for p in (
        root / "policy" / "SPACE" / "block",
        root / "policy" / "SPACE" / "spacepolicy",
    ):
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)

def _quat_xyzw_to_rot(quat: Tensor) -> Tensor:
    q = F.normalize(quat, dim=-1)
    x, y, z, w = q.unbind(-1)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    row0 = torch.stack([1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)], dim=-1)
    row1 = torch.stack([2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)], dim=-1)
    row2 = torch.stack([2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)

def _pose7_to_T(pose7: Tensor) -> Tensor:
    B = pose7.shape[0]
    T = torch.eye(4, device=pose7.device, dtype=pose7.dtype).unsqueeze(0).repeat(B, 1, 1)
    T[:, :3, :3] = _quat_xyzw_to_rot(pose7[:, 3:7])
    T[:, :3, 3] = pose7[:, :3]
    return T

def _transform_points(T: Tensor, pts: Tensor) -> Tensor:
    return torch.einsum("bij,mj->bmi", T[:, :3, :3], pts) + T[:, :3, 3].unsqueeze(1)

def _invert_se3(T: Tensor) -> Tensor:
    R = T[:, :3, :3]
    t = T[:, :3, 3]
    R_inv = R.transpose(-1, -2)
    t_inv = -(R_inv @ t.unsqueeze(-1)).squeeze(-1)
    out = torch.zeros_like(T)
    out[:, :3, :3] = R_inv
    out[:, :3, 3] = t_inv
    out[:, 3, 3] = 1.0
    return out

def build_dual_panda_ee_proxy_robot_pc(
    agent_pos: Tensor, points_per_link: int = 16
) -> Dict[str, Tensor]:
    """Temporary DualPanda proxy PC from EE poses (no full Panda link FK yet).

    RLBench2 PPI uses DualPanda. Until DualPanda URDF FK / precomputed PC is
    wired, sample virtual wrist/finger clouds in each Panda EE frame.

    agent_pos: [B,16] = left_pose7 + right_pose7 + left_open + right_open
    quat order: xyzw (RLBench / PyRep gripper_pose).
    """
    if agent_pos.ndim != 2 or agent_pos.shape[-1] < 14:
        raise ValueError(f"agent_pos expected [B,16], got {tuple(agent_pos.shape)}")
    warnings.warn(
        "[PPI_rp1] DualPanda EE-proxy robot PC (not full Panda link FK). "
        "Prefer precomputed DualPanda left/right geometry+topology PCs.",
        stacklevel=2,
    )
    B = agent_pos.shape[0]
    device, dtype = agent_pos.device, agent_pos.dtype
    M = int(points_per_link)

    left_T = _pose7_to_T(agent_pos[:, 0:7])
    right_T = _pose7_to_T(agent_pos[:, 7:14])

    gen = torch.Generator(device="cpu")
    gen.manual_seed(0)
    wrist_local = ((torch.rand(M, 3, generator=gen) - 0.5) * 0.06).to(device=device, dtype=dtype)
    finger_local = ((torch.rand(M, 3, generator=gen) - 0.5) * 0.04).to(device=device, dtype=dtype)
    finger_local = finger_local.clone()
    finger_local[:, 2] += 0.04

    def one_arm(T_ee: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        z = T_ee[:, :3, 2]
        T_finger = T_ee.clone()
        T_finger[:, :3, 3] = T_ee[:, :3, 3] + 0.04 * z
        p_w = _transform_points(T_ee, wrist_local)
        p_f = _transform_points(T_finger, finger_local)
        geo = torch.cat([p_w, p_f], dim=1)
        rels = []
        for sid, nid, p_world, T_nei in (
            (0, 1, p_w, T_finger),
            (1, 0, p_f, T_ee),
        ):
            T_inv = _invert_se3(T_nei)
            ones = torch.ones(B, M, 1, device=device, dtype=dtype)
            p_h = torch.cat([p_world, ones], dim=-1)
            p_in_j = torch.einsum("bij,bmj->bmi", T_inv, p_h)[..., :3]
            sid_c = torch.full((B, M, 1), float(sid), device=device, dtype=dtype)
            nid_c = torch.full((B, M, 1), float(nid), device=device, dtype=dtype)
            rels.append(torch.cat([p_in_j, sid_c, nid_c], dim=-1))
        topo = torch.stack(rels, dim=1)
        link_means = torch.stack([p_w.mean(1), p_f.mean(1)], dim=1)
        return geo, topo, link_means

    l_geo, l_topo, l_means = one_arm(left_T)
    r_geo, r_topo, r_means = one_arm(right_T)
    return {
        "left_geometry_pc": l_geo,
        "right_geometry_pc": r_geo,
        "left_topology_pc": l_topo,
        "right_topology_pc": r_topo,
        "left_link_means": l_means,
        "right_link_means": r_means,
    }

class RobotGeoTopoTokenEncoder(nn.Module):
    """Encode left/right geo + topo features as tokens (no fusion).

    Returns:
      robot_feat  [B, Nr, token_dim]
      robot_coord [B, Nr, 3]
    """

    def __init__(
        self,
        token_dim: int = 288,
        robot_dim: int = 128,
        points_per_link: int = 16,
        use_geometry: bool = True,
        use_topology: bool = True,
        pc_backbone: str = "pointnet",
        use_type_embed: bool = True,
        allow_ee_proxy: bool = True,
        # RLBench2 PPI = DualPanda; do NOT default to RobotWin AgileX FK.
        robot_embodiment: str = "dual_panda",
        debug: bool = False,
    ):
        super().__init__()
        if not use_geometry and not use_topology:
            raise ValueError("need at least one of use_geometry / use_topology")
        self.token_dim = int(token_dim)
        self.robot_dim = int(robot_dim)
        self.points_per_link = int(points_per_link)
        self.use_geometry = bool(use_geometry)
        self.use_topology = bool(use_topology)
        self.allow_ee_proxy = bool(allow_ee_proxy)
        self.robot_embodiment = str(robot_embodiment).lower().strip()
        self.debug = bool(debug)

        _ensure_dp3_paths()
        from backbone.model.vision.pointnet_extractor import (
            build_robot_geometry_encoder,
            build_robot_topology_encoder,
        )

        self.left_geometry_encoder = None
        self.right_geometry_encoder = None
        self.left_topology_encoder = None
        self.right_topology_encoder = None
        if self.use_geometry:
            cfg = dict(
                backbone=pc_backbone,
                in_channels=3,
                out_channels=robot_dim,
                use_layernorm=True,
                final_norm="layernorm",
            )
            self.left_geometry_encoder = build_robot_geometry_encoder(**cfg)
            self.right_geometry_encoder = build_robot_geometry_encoder(**cfg)
        if self.use_topology:
            cfg = dict(
                backbone=pc_backbone,
                in_channels=5,
                out_channels=robot_dim,
                use_layernorm=True,
                final_norm="layernorm",
            )
            self.left_topology_encoder = build_robot_topology_encoder(**cfg)
            self.right_topology_encoder = build_robot_topology_encoder(**cfg)

        self.proj = nn.Sequential(nn.Linear(robot_dim, token_dim), nn.LayerNorm(token_dim))
        # 0=left_geo, 1=left_topo, 2=right_geo, 3=right_topo
        self.type_embed = nn.Embedding(4, token_dim) if use_type_embed else None

    def _device(self) -> torch.device:
        return next(self.parameters()).device

    def _encode_topology_group(self, topology_pc: Tensor, encoder: nn.Module) -> List[Tensor]:
        if topology_pc.ndim != 4 or topology_pc.shape[-1] != 5:
            raise ValueError(f"topology_pc expected [B,K,M,5], got {tuple(topology_pc.shape)}")
        B, K, M, _ = topology_pc.shape
        feat = encoder(topology_pc.reshape(B * K, M, 5)).reshape(B, K, -1)
        return [feat[:, k, :] for k in range(K)]

    def _link_means_from_geometry(self, geo: Tensor, n_links: int, m: int) -> Tensor:
        B = geo.shape[0]
        return geo.reshape(B, n_links, m, 3).mean(dim=2)

    def _resolve_robot_pc(self, observations: Dict) -> Dict[str, Tensor]:
        device = self._device()

        # 1) precomputed DualPanda geo/topo PCs — preferred for RLBench2
        if all(k in observations for k in ROBOT_PC_KEYS):
            pc = {
                k: torch.as_tensor(observations[k], device=device, dtype=torch.float32)
                for k in ROBOT_PC_KEYS
            }
            for side in ("left", "right"):
                key = f"{side}_link_means"
                if key in observations:
                    pc[key] = torch.as_tensor(
                        observations[key], device=device, dtype=torch.float32
                    )
            return pc

        # 2) DualPanda (default): never call AgileX make_robot.
        #    Full Panda link FK not wired yet → EE-proxy or precomputed PC only.
        if self.robot_embodiment in ("dual_panda", "panda", "rlbench2"):
            if observations.get("robot_qpos", None) is not None and self.debug:
                print(
                    "[RobotGeoTopoTokenEncoder] dual_panda: ignoring robot_qpos for PC; "
                    "use precomputed DualPanda robot_pc or EE-proxy."
                )
            if self.allow_ee_proxy:
                agent = observations.get("robot_state_raw", observations.get("agent_pos"))
                if agent is None:
                    raise KeyError(
                        "dual_panda needs precomputed robot_pc or "
                        "robot_state_raw/agent_pos[B,16] for EE-proxy"
                    )
                agent = torch.as_tensor(agent, dtype=torch.float32, device=device)
                if agent.ndim == 1:
                    agent = agent.unsqueeze(0)
                return build_dual_panda_ee_proxy_robot_pc(
                    agent, points_per_link=self.points_per_link
                )
            raise KeyError(
                "dual_panda requires precomputed left/right geometry+topology PCs "
                "(or robot_allow_ee_proxy=true with agent_pos)"
            )

        # 3) RobotWin AgileX only when explicitly requested
        if self.robot_embodiment in ("agilex", "aloha", "robotwin"):
            qpos = observations.get("robot_qpos", observations.get("agent_pos_raw", None))
            if qpos is not None:
                q = torch.as_tensor(qpos, dtype=torch.float32, device=device)
                if q.ndim == 1:
                    q = q.unsqueeze(0)
                if q.shape[-1] != 14:
                    raise ValueError(
                        f"AgileX robot_qpos expected [B,14], got {tuple(q.shape)}"
                    )
                _ensure_dp3_paths()
                from make_robot import RobotPointCloudGenerator

                gen = RobotPointCloudGenerator(
                    points_per_link=self.points_per_link, device=device
                )
                online = gen.make_robot(q)
                if online["left_geometry_pc"].ndim == 2:
                    online = {k: v.unsqueeze(0) for k, v in online.items()}
                online["left_link_means"] = self._link_means_from_geometry(
                    online["left_geometry_pc"], gen.n_links_per_arm, gen.M
                )
                online["right_link_means"] = self._link_means_from_geometry(
                    online["right_geometry_pc"], gen.n_links_per_arm, gen.M
                )
                return online

        raise KeyError(
            f"Cannot resolve robot PC for embodiment={self.robot_embodiment!r}. "
            "For DualPanda provide precomputed robot_pc or enable EE-proxy."
        )

    def forward(self, observations: Dict) -> Tuple[Tensor, Tensor]:
        pc = self._resolve_robot_pc(observations)
        left_geo = pc["left_geometry_pc"]
        right_geo = pc["right_geometry_pc"]
        if left_geo.ndim == 2:
            left_geo = left_geo.unsqueeze(0)
            right_geo = right_geo.unsqueeze(0)

        left_means = pc.get("left_link_means")
        right_means = pc.get("right_link_means")
        if left_means is None:
            left_means = left_geo.mean(dim=1, keepdim=True)
            right_means = right_geo.mean(dim=1, keepdim=True)
        if left_means.ndim == 2:
            left_means = left_means.unsqueeze(1)
            right_means = right_means.unsqueeze(1)

        feats: List[Tensor] = []
        coords: List[Tensor] = []
        types: List[int] = []

        def append_arm(
            geo_pc: Tensor,
            geo_enc: Optional[nn.Module],
            topo_pc: Optional[Tensor],
            topo_enc: Optional[nn.Module],
            link_means: Tensor,
            geo_type: int,
            topo_type: int,
        ):
            if self.use_geometry:
                assert geo_enc is not None
                feats.append(geo_enc(geo_pc))
                coords.append(geo_pc.mean(dim=1))
                types.append(geo_type)
            if self.use_topology:
                assert topo_pc is not None and topo_enc is not None
                if topo_pc.ndim == 3:
                    topo_pc = topo_pc.unsqueeze(0)
                for k, ft in enumerate(self._encode_topology_group(topo_pc, topo_enc)):
                    feats.append(ft)
                    li = int(topo_pc[0, k, 0, 3].item()) if topo_pc.shape[-1] >= 4 else 0
                    li = min(max(li, 0), link_means.shape[1] - 1)
                    coords.append(link_means[:, li, :])
                    types.append(topo_type)

        append_arm(
            left_geo,
            self.left_geometry_encoder,
            pc.get("left_topology_pc"),
            self.left_topology_encoder,
            left_means,
            geo_type=0,
            topo_type=1,
        )
        append_arm(
            right_geo,
            self.right_geometry_encoder,
            pc.get("right_topology_pc"),
            self.right_topology_encoder,
            right_means,
            geo_type=2,
            topo_type=3,
        )

        robot_feat = torch.stack(feats, dim=1)
        robot_coord = torch.stack(coords, dim=1)
        robot_feat = self.proj(robot_feat)
        if self.type_embed is not None:
            type_ids = torch.tensor(types, device=robot_feat.device, dtype=torch.long)
            robot_feat = robot_feat + self.type_embed(type_ids)[None]

        if self.debug:
            print(f"[RobotGeoTopoTokenEncoder] feat={tuple(robot_feat.shape)} coord={tuple(robot_coord.shape)}")
        return robot_feat, robot_coord
