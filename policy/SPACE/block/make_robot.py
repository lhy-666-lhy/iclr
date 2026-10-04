"""Robot explicit representation: qpos → geometry / topology point clouds.

Standalone module for SPACE robot-aware pipeline. No neural networks.
Does not modify any existing SPACE training files.

qpos layout (RoboTwin / AgileX, 14-D):
    [left_j1..left_j6, left_gripper, right_j1..right_j6, right_gripper]
"""

from __future__ import annotations

import hashlib
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import trimesh
import yaml

Tensor = torch.Tensor
PathLike = Union[str, Path]

# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------

def _parse_floats(text: str) -> List[float]:
    return [float(x) for x in text.split()]

def _stable_link_seed(name: str) -> int:
    digest = hashlib.md5(name.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % (2**31)

def _rpy_matrix(rpy: Sequence[float], device: torch.device, dtype: torch.dtype) -> Tensor:
    roll, pitch, yaw = [torch.tensor(v, device=device, dtype=dtype) for v in rpy]
    cr, sr = torch.cos(roll), torch.sin(roll)
    cp, sp = torch.cos(pitch), torch.sin(pitch)
    cy, sy = torch.cos(yaw), torch.sin(yaw)
    z = torch.tensor(0.0, device=device, dtype=dtype)
    o = torch.tensor(1.0, device=device, dtype=dtype)
    rx = torch.stack(
        [
            torch.stack([o, z, z]),
            torch.stack([z, cr, -sr]),
            torch.stack([z, sr, cr]),
        ]
    )
    ry = torch.stack(
        [
            torch.stack([cp, z, sp]),
            torch.stack([z, o, z]),
            torch.stack([-sp, z, cp]),
        ]
    )
    rz = torch.stack(
        [
            torch.stack([cy, -sy, z]),
            torch.stack([sy, cy, z]),
            torch.stack([z, z, o]),
        ]
    )
    return rz @ ry @ rx

def _origin_matrix(
    xyz: Sequence[float], rpy: Sequence[float], device: torch.device, dtype: torch.dtype
) -> Tensor:
    mat = torch.eye(4, device=device, dtype=dtype)
    mat[:3, :3] = _rpy_matrix(rpy, device, dtype)
    mat[:3, 3] = torch.tensor(list(xyz), device=device, dtype=dtype)
    return mat

def _axis_angle_matrix(axis: Tensor, angle: Tensor) -> Tensor:
    """Rodrigues → SE(3). axis (3,), angle (B,) → (B, 4, 4)."""
    axis = F.normalize(axis, dim=-1)
    x, y, z = axis[0], axis[1], axis[2]
    c, s = torch.cos(angle), torch.sin(angle)
    one_c = 1.0 - c
    r00 = c + x * x * one_c
    r01 = x * y * one_c - z * s
    r02 = x * z * one_c + y * s
    r10 = y * x * one_c + z * s
    r11 = c + y * y * one_c
    r12 = y * z * one_c - x * s
    r20 = z * x * one_c - y * s
    r21 = z * y * one_c + x * s
    r22 = c + z * z * one_c
    row0 = torch.stack([r00, r01, r02], dim=-1)
    row1 = torch.stack([r10, r11, r12], dim=-1)
    row2 = torch.stack([r20, r21, r22], dim=-1)
    rot = torch.stack([row0, row1, row2], dim=-2)
    mat = torch.zeros(angle.shape[0], 4, 4, device=angle.device, dtype=angle.dtype)
    mat[:, :3, :3] = rot
    mat[:, 3, 3] = 1.0
    return mat

def _prismatic_matrix(axis: Tensor, dist: Tensor) -> Tensor:
    axis = F.normalize(axis, dim=-1)
    mat = torch.zeros(dist.shape[0], 4, 4, device=dist.device, dtype=dist.dtype)
    mat[:, :3, :3] = torch.eye(3, device=dist.device, dtype=dist.dtype)
    mat[:, :3, 3] = axis.unsqueeze(0) * dist.unsqueeze(-1)
    mat[:, 3, 3] = 1.0
    return mat

def _quat_pose_matrix(pose7: Sequence[float]) -> np.ndarray:
    """[x, y, z, qw, qx, qy, qz] → 4x4."""
    xyz = pose7[:3]
    qw, qx, qy, qz = pose7[3:]
    xx, yy, zz = qx * qx, qy * qy, qz * qz
    xy, xz, yz = qx * qy, qx * qz, qy * qz
    wx, wy, wz = qw * qx, qw * qy, qw * qz
    rot = np.array(
        [
            [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
        ],
        dtype=np.float64,
    )
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = rot
    mat[:3, 3] = xyz
    return mat

def _np_origin_matrix(xyz: np.ndarray, rpy: np.ndarray) -> np.ndarray:
    mat = np.eye(4, dtype=np.float64)
    cr, sr = np.cos(rpy[0]), np.sin(rpy[0])
    cp, sp = np.cos(rpy[1]), np.sin(rpy[1])
    cy, sy = np.cos(rpy[2]), np.sin(rpy[2])
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    mat[:3, :3] = rz @ ry @ rx
    mat[:3, 3] = xyz
    return mat

def _invert_se3(T: Tensor) -> Tensor:
    """Invert (B,4,4) or (4,4) rigid transforms."""
    single = T.ndim == 2
    if single:
        T = T.unsqueeze(0)
    R = T[:, :3, :3]
    t = T[:, :3, 3]
    R_inv = R.transpose(-1, -2)
    t_inv = -(R_inv @ t.unsqueeze(-1)).squeeze(-1)
    out = torch.zeros_like(T)
    out[:, :3, :3] = R_inv
    out[:, :3, 3] = t_inv
    out[:, 3, 3] = 1.0
    return out[0] if single else out

def _transform_local_points(T: Tensor, pts: Tensor) -> Tensor:
    """T: (B,4,4), pts: (M,3) → (B,M,3)."""
    return torch.einsum("bij,mj->bmi", T[:, :3, :3], pts) + T[:, :3, 3].unsqueeze(1)

# ---------------------------------------------------------------------------
# URDF / mesh loading
# ---------------------------------------------------------------------------

def _default_asset_dir() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        cand = parent / "assets" / "embodiments" / "aloha-agilex"
        if cand.exists():
            return cand
    raise FileNotFoundError(
        "Cannot find assets/embodiments/aloha-agilex. Pass urdf_path / mesh_path / robot_config."
    )

def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f.read()) or {}

def _parse_gripper_joints(gripper_cfg: dict) -> Dict[str, Dict[str, float]]:
    joints: Dict[str, Dict[str, float]] = {}
    if not gripper_cfg:
        return joints
    base = gripper_cfg["base"]
    joints[base] = {"mult": 1.0, "offset": 0.0}
    for mimic_name, mult, offset in gripper_cfg.get("mimic", []):
        joints[mimic_name] = {"mult": float(mult), "offset": float(offset)}
    return joints

def _load_collision_trimesh(link_el: ET.Element, urdf_dir: Path) -> Optional[trimesh.Trimesh]:
    block = link_el.find("collision")
    if block is None:
        return None
    origin_el = block.find("origin")
    xyz = (
        np.array(_parse_floats(origin_el.get("xyz", "0 0 0")), dtype=np.float64)
        if origin_el is not None
        else np.zeros(3)
    )
    rpy = (
        np.array(_parse_floats(origin_el.get("rpy", "0 0 0")), dtype=np.float64)
        if origin_el is not None
        else np.zeros(3)
    )
    local_tf = _np_origin_matrix(xyz, rpy)

    geom = block.find("geometry")
    if geom is None:
        return None

    mesh = None
    mesh_el = geom.find("mesh")
    if mesh_el is not None:
        path = (urdf_dir / mesh_el.get("filename")).resolve()
        if path.exists():
            mesh = trimesh.load(str(path), force="mesh")

    box_el = geom.find("box")
    if mesh is None and box_el is not None:
        size = np.array(_parse_floats(box_el.get("size")), dtype=np.float64)
        mesh = trimesh.creation.box(extents=size)

    cyl_el = geom.find("cylinder")
    if mesh is None and cyl_el is not None:
        mesh = trimesh.creation.cylinder(
            radius=float(cyl_el.get("radius")),
            height=float(cyl_el.get("length")),
        )

    if mesh is None:
        return None
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    mesh = mesh.copy()
    mesh.apply_transform(local_tf)
    return mesh

def _sample_mesh_surface(mesh: Optional[trimesh.Trimesh], n_points: int, seed: int = 0) -> np.ndarray:
    if mesh is None or len(mesh.vertices) == 0:
        return np.zeros((n_points, 3), dtype=np.float32)
    seed = int(seed) % (2**31)
    try:
        surf, _ = trimesh.sample.sample_surface(mesh, n_points, seed=seed)
    except TypeError:
        state = np.random.get_state()
        np.random.seed(seed)
        try:
            surf, _ = trimesh.sample.sample_surface(mesh, n_points)
        finally:
            np.random.set_state(state)
    return np.asarray(surf, dtype=np.float32)

def _sample_mesh_inner(mesh: Optional[trimesh.Trimesh], n_points: int, seed: int = 0) -> np.ndarray:
    """Sample points inside mesh volume (occupied interior), not only surface.

    Strategy (no surface fallback — failures raise):
      1) trimesh volume samplers if available / mesh is usable
      2) AABB uniform proposal + mesh.contains rejection
    """
    if mesh is None or len(mesh.vertices) == 0:
        raise RuntimeError("inner sampling failed: mesh is None or empty")
    n_points = int(n_points)
    seed = int(seed) % (2**31)
    rng = np.random.default_rng(seed)
    errors: List[str] = []

    # 1) Prefer library volume sampler
    if hasattr(trimesh.sample, "volume_mesh"):
        try:
            pts = trimesh.sample.volume_mesh(mesh, n_points)
            pts = np.asarray(pts, dtype=np.float32)
            if pts.ndim == 2 and pts.shape[0] == n_points and pts.shape[1] == 3:
                return pts
            errors.append(
                f"volume_mesh returned shape {getattr(pts, 'shape', None)}, need ({n_points}, 3)"
            )
        except Exception as e:
            errors.append(f"volume_mesh: {type(e).__name__}: {e}")
    if hasattr(mesh, "sample_volume"):
        try:
            pts = mesh.sample_volume(n_points)
            pts = np.asarray(pts, dtype=np.float32)
            if pts.ndim == 2 and pts.shape[0] >= n_points:
                return pts[:n_points]
            errors.append(
                f"sample_volume returned shape {getattr(pts, 'shape', None)}, need >= ({n_points}, 3)"
            )
        except Exception as e:
            errors.append(f"sample_volume: {type(e).__name__}: {e}")

    # 2) Rejection sampling inside AABB
    try:
        bounds = np.asarray(mesh.bounds, dtype=np.float64)
        lo, hi = bounds[0], bounds[1]
        if not np.all(np.isfinite(lo)) or not np.all(np.isfinite(hi)):
            raise RuntimeError(f"invalid mesh bounds: {bounds}")
        collected: List[np.ndarray] = []
        n_have = 0
        max_batches = 64
        batch = max(n_points * 16, 512)
        contains_err: Optional[str] = None
        for _ in range(max_batches):
            cand = rng.uniform(lo, hi, size=(batch, 3))
            try:
                inside = mesh.contains(cand)
            except Exception as e:
                contains_err = f"mesh.contains: {type(e).__name__}: {e}"
                break
            hit = cand[np.asarray(inside, dtype=bool)]
            if hit.size == 0:
                continue
            take = min(n_points - n_have, hit.shape[0])
            collected.append(hit[:take].astype(np.float32))
            n_have += take
            if n_have >= n_points:
                return np.concatenate(collected, axis=0)[:n_points]
        if contains_err is not None:
            errors.append(contains_err)
        else:
            errors.append(
                f"AABB rejection got {n_have}/{n_points} points "
                f"(watertight={getattr(mesh, 'is_watertight', None)}, "
                f"volume={getattr(mesh, 'volume', None)})"
            )
    except Exception as e:
        errors.append(f"AABB rejection: {type(e).__name__}: {e}")

    detail = "; ".join(errors) if errors else "unknown reason"
    raise RuntimeError(
        f"inner (volume) sampling failed for mesh "
        f"(V={len(mesh.vertices)}, F={len(getattr(mesh, 'faces', []))}); "
        f"no surface fallback. details: {detail}"
    )

def _sample_mesh_points(
    mesh: Optional[trimesh.Trimesh],
    n_points: int,
    seed: int = 0,
    region: str = "surface",
) -> np.ndarray:
    region = str(region).lower().strip()
    if region in ("surface", "surf", "shell"):
        return _sample_mesh_surface(mesh, n_points, seed=seed)
    if region in ("inner", "volume", "interior", "inside"):
        return _sample_mesh_inner(mesh, n_points, seed=seed)
    raise ValueError(f"sample region must be 'surface' or 'inner', got {region!r}")

def _discover_arm_links(urdf_path: Path, prefix: str) -> Tuple[List[str], List[Optional[trimesh.Trimesh]]]:
    root = ET.parse(urdf_path).getroot()
    urdf_dir = urdf_path.parent
    link_names, meshes = [], []
    for link in root.findall("link"):
        name = link.get("name")
        if not name.startswith(prefix):
            continue
        if any(k in name for k in ("camera", "wheel", "castor")):
            continue
        mesh = _load_collision_trimesh(link, urdf_dir)
        if mesh is None:
            continue
        link_names.append(name)
        meshes.append(mesh)
    if not link_names:
        raise RuntimeError(f"No collision meshes for prefix {prefix} in {urdf_path}")
    return link_names, meshes

def _parse_arm_joints(urdf_path: Path, prefix: str) -> Dict[str, Dict[str, Any]]:
    root = ET.parse(urdf_path).getroot()
    joints = {}
    for joint in root.findall("joint"):
        name = joint.get("name")
        if not name or not name.startswith(prefix):
            continue
        origin = joint.find("origin")
        axis = joint.find("axis")
        child = joint.find("child")
        parent = joint.find("parent")
        if child is None or parent is None:
            continue
        joints[child.get("link")] = {
            "name": name,
            "parent": parent.get("link"),
            "xyz": _parse_floats(origin.get("xyz", "0 0 0")) if origin is not None else [0, 0, 0],
            "rpy": _parse_floats(origin.get("rpy", "0 0 0")) if origin is not None else [0, 0, 0],
            "axis": _parse_floats(axis.get("xyz", "0 0 1")) if axis is not None else [0, 0, 1],
            "type": joint.get("type"),
        }
    return joints

def _topo_sort_links(link_names: List[str], joint_meta: Dict[str, Dict[str, Any]], base_link: str) -> List[str]:
    pending = set(link_names)
    if base_link not in pending:
        raise RuntimeError(f"base link {base_link} missing from {link_names}")
    order = [base_link]
    pending.remove(base_link)
    done = {base_link}
    while pending:
        progressed = False
        for child in list(pending):
            meta = joint_meta.get(child)
            if meta is None:
                pending.remove(child)
                continue
            parent = meta["parent"]
            if parent in done:
                order.append(child)
                pending.remove(child)
                done.add(child)
                progressed = True
        if not progressed:
            # orphan (e.g. parent outside arm): attach after base
            for child in list(pending):
                order.append(child)
                pending.remove(child)
                done.add(child)
            break
    return order

def _chain_neighbor_pairs(n_links: int) -> List[Tuple[int, int]]:
    """Directed neighbor pairs along a pure chain (legacy helper)."""
    pairs: List[Tuple[int, int]] = []
    for i in range(n_links - 1):
        pairs.append((i, i + 1))
        pairs.append((i + 1, i))
    return pairs

def _arm_topology_pairs(
    link_names: List[str],
    joint_meta: Dict[str, Dict[str, Any]],
    prefix: str,
) -> List[Tuple[int, int]]:
    """Directed topology edges for one arm (no TCP).

    Includes:
      - URDF parent–child edges among arm links (base↔1↔…↔6, and 6↔7, 6↔8)
      - gripper finger adjacency 7↔8
    Each undirected edge → two directed pairs.
    """
    name_to_idx = {n: i for i, n in enumerate(link_names)}
    undirected: set = set()
    for child, meta in joint_meta.items():
        if not child.startswith(prefix) or child not in name_to_idx:
            continue
        parent = meta["parent"]
        if parent not in name_to_idx:
            continue
        a, b = name_to_idx[parent], name_to_idx[child]
        undirected.add((min(a, b), max(a, b)))

    # Explicit finger–finger adjacency (siblings under link6)
    l7, l8 = f"{prefix}link7", f"{prefix}link8"
    if l7 in name_to_idx and l8 in name_to_idx:
        a, b = name_to_idx[l7], name_to_idx[l8]
        undirected.add((min(a, b), max(a, b)))

    pairs: List[Tuple[int, int]] = []
    for a, b in sorted(undirected):
        pairs.append((a, b))
        pairs.append((b, a))
    return pairs

def _preferred_arm_link_order(mesh_names: List[str], prefix: str) -> List[str]:
    """Stable order: base, link1..link6, link7, link8 (drop unknown extras at end)."""
    preferred = [f"{prefix}base_link"] + [f"{prefix}link{i}" for i in range(1, 9)]
    have = set(mesh_names)
    ordered = [n for n in preferred if n in have]
    # any leftover collision links
    for n in mesh_names:
        if n not in ordered:
            ordered.append(n)
    return ordered

# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

class RobotPointCloudGenerator:
    """qpos → left/right geometry + topology point clouds (no NN)."""

    def __init__(
        self,
        urdf_path: Optional[PathLike] = None,
        mesh_path: Optional[PathLike] = None,
        robot_config: Optional[Union[PathLike, dict]] = None,
        points_per_link: int = 64,
        points_per_arm: Optional[int] = None,
        sample_region: str = "surface",
        device: Union[str, torch.device] = "cpu",
    ):
        """
        Args:
            urdf_path: path to arx5 URDF (default: RoboTwin aloha-agilex).
            mesh_path: URDF mesh root (default: urdf parent dir).
            robot_config: embodiment config.yml path or dict.
            points_per_link: local samples per link (geometry + topology M).
            points_per_arm: if set, redistribute total points across links.
            sample_region: "surface" (trimesh surface) or "inner" (volume/occupancy).
            device: torch device for buffers / FK.
        """
        self.device = torch.device(device)
        region = str(sample_region).lower().strip()
        if region in ("surf", "shell"):
            region = "surface"
        if region in ("volume", "interior", "inside"):
            region = "inner"
        if region not in ("surface", "inner"):
            raise ValueError(f"sample_region must be 'surface' or 'inner', got {sample_region!r}")
        self.sample_region = region
        need_default_asset = urdf_path is None or robot_config is None
        asset_dir = _default_asset_dir() if need_default_asset else None
        if urdf_path is None:
            urdf_path = asset_dir / "urdf" / "arx5_description_isaac.urdf"
        self.urdf_path = Path(urdf_path)
        self.mesh_path = Path(mesh_path) if mesh_path is not None else self.urdf_path.parent

        if robot_config is None:
            robot_config = asset_dir / "config.yml"
        if isinstance(robot_config, (str, Path)):
            self.robot_config = _load_yaml(Path(robot_config))
        else:
            self.robot_config = dict(robot_config or {})

        gripper_scale = self.robot_config.get("gripper_scale", [-0.01, 0.045])
        self.gripper_scale = torch.tensor(gripper_scale, dtype=torch.float32, device=self.device)

        gripper_cfgs = self.robot_config.get("gripper_name", [{}, {}])
        self.left_gripper_joints = _parse_gripper_joints(gripper_cfgs[0] if gripper_cfgs else {})
        self.right_gripper_joints = _parse_gripper_joints(
            gripper_cfgs[1] if len(gripper_cfgs) > 1 else {}
        )

        robot_pose = self.robot_config.get("robot_pose", [[0, -0.65, 0.0, 0.707, 0, 0, 0.707]])[0]
        self.world_root = torch.from_numpy(_quat_pose_matrix(robot_pose)).float().to(self.device)

        self.left_prefix = "fl_"
        self.right_prefix = "fr_"
        self.joint_meta = {
            **_parse_arm_joints(self.urdf_path, self.left_prefix),
            **_parse_arm_joints(self.urdf_path, self.right_prefix),
        }

        left_mesh_names, left_meshes = _discover_arm_links(self.urdf_path, self.left_prefix)
        right_mesh_names, right_meshes = _discover_arm_links(self.urdf_path, self.right_prefix)

        # Physical links only (no virtual TCP): base + link1..6 + two gripper fingers
        self.left_links = _preferred_arm_link_order(left_mesh_names, self.left_prefix)
        self.right_links = _preferred_arm_link_order(right_mesh_names, self.right_prefix)

        self.n_links_per_arm = len(self.left_links)
        assert len(self.right_links) == self.n_links_per_arm

        mesh_map_l = dict(zip(left_mesh_names, left_meshes))
        mesh_map_r = dict(zip(right_mesh_names, right_meshes))

        if points_per_arm is not None:
            self.points_per_link = max(1, int(points_per_arm) // self.n_links_per_arm)
            rem = int(points_per_arm) - self.points_per_link * self.n_links_per_arm
        else:
            self.points_per_link = int(points_per_link)
            rem = 0

        self.M = self.points_per_link

        self.left_local, self.left_sizes = self._sample_local_clouds(
            self.left_links, mesh_map_l, rem
        )
        self.right_local, self.right_sizes = self._sample_local_clouds(
            self.right_links, mesh_map_r, rem
        )

        self.left_q_map = {f"{self.left_prefix}joint{i}": i - 1 for i in range(1, 7)}
        self.right_q_map = {f"{self.right_prefix}joint{i}": i - 1 for i in range(1, 7)}

        left_base_key = f"{self.left_prefix}base_link"
        right_base_key = f"{self.right_prefix}base_link"
        if left_base_key not in self.joint_meta or right_base_key not in self.joint_meta:
            raise RuntimeError(
                f"Missing base joint meta for {left_base_key}/{right_base_key}. "
                "Check URDF fl_base_joint / fr_base_joint."
            )
        self.left_base_origin = _origin_matrix(
            self.joint_meta[left_base_key]["xyz"],
            self.joint_meta[left_base_key]["rpy"],
            self.device,
            torch.float32,
        )
        self.right_base_origin = _origin_matrix(
            self.joint_meta[right_base_key]["xyz"],
            self.joint_meta[right_base_key]["rpy"],
            self.device,
            torch.float32,
        )

        # Topology: URDF tree edges + gripper 7↔8 (no TCP, no false 7–8 chain via serial order)
        self.left_neighbor_pairs = _arm_topology_pairs(
            self.left_links, self.joint_meta, self.left_prefix
        )
        self.right_neighbor_pairs = _arm_topology_pairs(
            self.right_links, self.joint_meta, self.right_prefix
        )
        # Back-compat alias (left); visualize/tests should prefer per-arm lists
        self.neighbor_pairs = self.left_neighbor_pairs
        self.n_topo_relations = len(self.left_neighbor_pairs)
        self.x = self.n_topo_relations // 2  # number of undirected edges

    def to(self, device: Union[str, torch.device]) -> "RobotPointCloudGenerator":
        """Move FK buffers / local clouds to device (not an nn.Module)."""
        device = torch.device(device)
        if device == self.device:
            return self
        self.device = device
        self.gripper_scale = self.gripper_scale.to(device)
        self.world_root = self.world_root.to(device)
        self.left_base_origin = self.left_base_origin.to(device)
        self.right_base_origin = self.right_base_origin.to(device)
        self.left_local = [p.to(device) for p in self.left_local]
        self.right_local = [p.to(device) for p in self.right_local]
        return self

    def _sample_local_clouds(
        self,
        link_names: List[str],
        mesh_map: Dict[str, Optional[trimesh.Trimesh]],
        rem: int,
    ) -> Tuple[List[Tensor], List[int]]:
        locals_list: List[Tensor] = []
        sizes: List[int] = []
        for i, name in enumerate(link_names):
            n_pts = self.points_per_link + (1 if rem > 0 and i >= len(link_names) - rem else 0)
            pts = _sample_mesh_points(
                mesh_map.get(name),
                n_pts,
                seed=_stable_link_seed(name),
                region=self.sample_region,
            )
            locals_list.append(torch.from_numpy(pts).float().to(self.device))
            sizes.append(n_pts)
        return locals_list, sizes

    # ----- FK -----

    def _resolve_joint_qpos(
        self,
        q_arm: Tensor,
        joint_name: str,
        q_map: Dict[str, int],
        gripper_joints: Dict[str, Dict[str, float]],
    ) -> Tensor:
        """q_arm: (B, 7) = 6 revolute + gripper."""
        if joint_name in gripper_joints:
            meta = gripper_joints[joint_name]
            lo, hi = self.gripper_scale[0], self.gripper_scale[1]
            # RoboTwin stores gripper in [0,1]-ish; map like ROG surface
            g = q_arm[:, 6]
            physical = lo + g * (hi - lo)
            return physical * meta["mult"] + meta["offset"]
        if joint_name in q_map:
            return q_arm[:, q_map[joint_name]]
        return torch.zeros(q_arm.shape[0], device=q_arm.device, dtype=q_arm.dtype)

    def forward_kinematics(
        self,
        qpos: Tensor,
        arm: str = "both",
    ) -> Dict[str, Dict[str, Tensor]]:
        """
        Compute world-frame SE(3) for each link.

        Args:
            qpos: (14,) or (B, 14)
            arm: 'left' | 'right' | 'both'

        Returns:
            {
              'left': {link_name: T (B,4,4), ...},
              'right': {...}
            }
        """
        q = torch.as_tensor(qpos, dtype=torch.float32, device=self.device)
        if q.ndim == 1:
            q = q.unsqueeze(0)
        if q.shape[-1] != 14:
            raise ValueError(f"qpos last dim must be 14, got {tuple(q.shape)}")

        out: Dict[str, Dict[str, Tensor]] = {}
        if arm in ("left", "both"):
            out["left"] = self._fk_arm(
                q[:, :7],
                self.left_links,
                self.left_q_map,
                self.left_base_origin,
                self.left_gripper_joints,
            )
        if arm in ("right", "both"):
            out["right"] = self._fk_arm(
                q[:, 7:],
                self.right_links,
                self.right_q_map,
                self.right_base_origin,
                self.right_gripper_joints,
            )
        return out

    def _fk_arm(
        self,
        q_arm: Tensor,
        link_names: List[str],
        q_map: Dict[str, int],
        base_origin: Tensor,
        gripper_joints: Dict[str, Dict[str, float]],
    ) -> Dict[str, Tensor]:
        B = q_arm.shape[0]
        device, dtype = q_arm.device, q_arm.dtype
        prefix = "fl_" if link_names[0].startswith("fl_") else "fr_"
        base_link = f"{prefix}base_link"

        transforms: Dict[str, Tensor] = {
            base_link: base_origin.to(device=device, dtype=dtype).unsqueeze(0).expand(B, -1, -1).contiguous(),
        }
        pending = {
            ln: self.joint_meta[ln]
            for ln in link_names
            if ln in self.joint_meta and ln != base_link
        }

        while pending:
            progressed = False
            for child, meta in list(pending.items()):
                parent = meta["parent"]
                if parent not in transforms:
                    continue
                origin = (
                    _origin_matrix(meta["xyz"], meta["rpy"], device, dtype)
                    .unsqueeze(0)
                    .expand(B, -1, -1)
                )
                axis = torch.tensor(meta["axis"], device=device, dtype=dtype)
                q_val = self._resolve_joint_qpos(q_arm, meta["name"], q_map, gripper_joints)
                if meta["type"] == "revolute":
                    motion = _axis_angle_matrix(axis, q_val)
                elif meta["type"] == "prismatic":
                    motion = _prismatic_matrix(axis, q_val)
                else:
                    motion = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).expand(B, -1, -1)
                transforms[child] = transforms[parent] @ origin @ motion
                del pending[child]
                progressed = True
            if not progressed:
                break

        root = self.world_root.to(device=device, dtype=dtype).unsqueeze(0).expand(B, -1, -1)
        return {name: root @ tf for name, tf in transforms.items()}

    # ----- Geometry -----

    def generate_geometry_pc(self, qpos: Tensor) -> Dict[str, Tensor]:
        """
        Returns:
            left_geometry_pc:  (N, 3)  or (B, N, 3)
            right_geometry_pc: (N, 3)  or (B, N, 3)
        """
        q = torch.as_tensor(qpos, dtype=torch.float32, device=self.device)
        squeeze = q.ndim == 1
        if squeeze:
            q = q.unsqueeze(0)

        fk = self.forward_kinematics(q)
        left = self._geometry_one_arm(fk["left"], self.left_links, self.left_local)
        right = self._geometry_one_arm(fk["right"], self.right_links, self.right_local)
        if squeeze:
            left, right = left[0], right[0]
        return {"left_geometry_pc": left, "right_geometry_pc": right}

    def _geometry_one_arm(
        self, transforms: Dict[str, Tensor], link_names: List[str], locals_list: List[Tensor]
    ) -> Tensor:
        clouds = []
        for name, pts_local in zip(link_names, locals_list):
            if name not in transforms:
                continue
            clouds.append(_transform_local_points(transforms[name], pts_local))
        return torch.cat(clouds, dim=1)  # (B, N, 3)

    # ----- Topology -----

    def generate_topology_pc(self, qpos: Tensor) -> Dict[str, Tensor]:
        """
        Topology: same world FK as geometry, then express self points in neighbor frame.

        Pipeline (per directed pair i → j):
          1) T_i, T_j = world SE(3) from forward_kinematics  (identical to geometry)
          2) p_world = T_i @ p_i_local                 # same as geometry world points
          3) p_in_j  = inv(T_j) @ p_world              # relative coordinates

        Equivalently:  p_in_j = inv(T_j) @ T_i @ p_i_local

        Returns:
            left/right_topology_pc: (2x, M, 5) or (B, 2x, M, 5)
            channels = [x, y, z, self_link_id, neighbor_link_id]  in neighbor frame
            x = #undirected edges (URDF tree + gripper 7↔8)
        """
        q = torch.as_tensor(qpos, dtype=torch.float32, device=self.device)
        squeeze = q.ndim == 1
        if squeeze:
            q = q.unsqueeze(0)

        # Same world FK used by generate_geometry_pc
        fk = self.forward_kinematics(q)
        left = self._topology_one_arm(
            fk["left"], self.left_links, self.left_local, self.left_neighbor_pairs
        )
        right = self._topology_one_arm(
            fk["right"], self.right_links, self.right_local, self.right_neighbor_pairs
        )
        if squeeze:
            left, right = left[0], right[0]
        return {"left_topology_pc": left, "right_topology_pc": right}

    def _topology_one_arm(
        self,
        transforms: Dict[str, Tensor],
        link_names: List[str],
        locals_list: List[Tensor],
        neighbor_pairs: List[Tuple[int, int]],
    ) -> Tensor:
        """transforms[*] must be world-frame (already include world_root)."""
        B = next(iter(transforms.values())).shape[0]
        device = self.device
        dtype = torch.float32
        rels = []
        for self_id, nei_id in neighbor_pairs:
            self_name = link_names[self_id]
            nei_name = link_names[nei_id]
            if self_name not in transforms or nei_name not in transforms:
                raise KeyError(f"Missing world T for {self_name} or {nei_name}")

            T_i = transforms[self_name]  # (B,4,4) world
            T_j = transforms[nei_name]  # (B,4,4) world
            p_local = locals_list[self_id]  # (M,3) in self link frame

            # Step 2: world points (same as geometry for this link)
            p_world = _transform_local_points(T_i, p_local)  # (B,M,3)

            # Step 3: world → neighbor frame
            # p_h = [x,y,z,1]; p_j = T_j^{-1} @ p_world_h
            T_j_inv = _invert_se3(T_j)
            ones = torch.ones(B, p_world.shape[1], 1, device=device, dtype=dtype)
            p_h = torch.cat([p_world, ones], dim=-1)  # (B,M,4)
            p_in_j = torch.einsum("bij,bmj->bmi", T_j_inv, p_h)[..., :3]

            M = p_in_j.shape[1]
            self_col = torch.full((B, M, 1), float(self_id), device=device, dtype=dtype)
            nei_col = torch.full((B, M, 1), float(nei_id), device=device, dtype=dtype)
            rels.append(torch.cat([p_in_j, self_col, nei_col], dim=-1))

        return torch.stack(rels, dim=1)  # (B, 2x, M, 5)  where 2x = len(neighbor_pairs)

    # ----- Public API -----

    def make_robot(self, qpos: Tensor) -> Dict[str, Tensor]:
        """
        Args:
            qpos: (14,) or (B, 14)

        Returns:
            left_geometry_pc:  (N,3) or (B,N,3)
            right_geometry_pc: (N,3) or (B,N,3)
            left_topology_pc:  (2x,M,5) or (B,2x,M,5)
            right_topology_pc: (2x,M,5) or (B,2x,M,5)
        """
        geo = self.generate_geometry_pc(qpos)
        topo = self.generate_topology_pc(qpos)
        return {**geo, **topo}

def make_robot(
    qpos: Union[Tensor, np.ndarray, Sequence[float]],
    generator: Optional[RobotPointCloudGenerator] = None,
    **kwargs,
) -> Dict[str, Tensor]:
    """Functional wrapper."""
    if generator is None:
        generator = RobotPointCloudGenerator(**kwargs)
    return generator.make_robot(torch.as_tensor(qpos, dtype=torch.float32))

# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _run_self_test() -> None:
    gen = RobotPointCloudGenerator(points_per_link=32, device="cpu")
    qpos = torch.randn(14) * 0.3
    out = gen.make_robot(qpos)

    lg, rg = out["left_geometry_pc"], out["right_geometry_pc"]
    lt, rt = out["left_topology_pc"], out["right_topology_pc"]

    print("=== make_robot self-test ===")
    print(f"links = {gen.left_links}")
    print(f"n_links = {gen.n_links_per_arm}, undirected_edges x = {gen.x}, directed 2x = {2 * gen.x}, M = {gen.M}")
    print(f"left_geometry_pc.shape  = {tuple(lg.shape)}  (expect [N,3])")
    print(f"right_geometry_pc.shape = {tuple(rg.shape)}  (expect [N,3])")
    print(f"left_topology_pc.shape  = {tuple(lt.shape)}  (expect [2x,M,5]=[{2*gen.x},{gen.M},5])")
    print(f"right_topology_pc.shape = {tuple(rt.shape)}  (expect [2x,M,5])")

    assert "TCP" not in "".join(gen.left_links + gen.right_links)
    assert lg.ndim == 2 and lg.shape[-1] == 3
    assert rg.ndim == 2 and rg.shape[-1] == 3
    assert lt.shape == (2 * gen.x, gen.M, 5)
    assert rt.shape == (2 * gen.x, gen.M, 5)

    # Terminal gripper topology must include 6↔7, 6↔8, 7↔8
    name_to_idx = {n: i for i, n in enumerate(gen.left_links)}
    i6, i7, i8 = name_to_idx["fl_link6"], name_to_idx["fl_link7"], name_to_idx["fl_link8"]
    undirected = {(min(a, b), max(a, b)) for a, b in gen.left_neighbor_pairs}
    for edge in ((i6, i7), (i6, i8), (i7, i8)):
        assert (min(edge), max(edge)) in undirected, f"missing edge {edge}"
    print("gripper edges OK: 6↔7, 6↔8, 7↔8")

    ids = lt[..., 3:5].reshape(-1, 2)
    assert ids.min() >= 0 and ids.max() < gen.n_links_per_arm
    print("link id range OK:", int(ids.min()), int(ids.max()))
    print("PASS")

if __name__ == "__main__":
    _run_self_test()
