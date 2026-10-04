"""DualPanda full-link mesh → geometry / topology point clouds for PPI_rp1.

Each arm has 12 parts: base, link0–7, gripper, finger1, finger2.
Uses CoppeliaSim/PyRep FK (same as scripts/viz/viz_dual_panda_robot_pc.py).
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

# Kinematic order (link indices 0..11)
DUAL_PANDA_LINK_NAMES: Tuple[str, ...] = (
    "base",
    "link0",
    "link1",
    "link2",
    "link3",
    "link4",
    "link5",
    "link6",
    "link7",
    "gripper",
    "finger1",
    "finger2",
)

N_LINKS_PER_ARM = len(DUAL_PANDA_LINK_NAMES)

ARM_SHAPES: Dict[str, List[Tuple[str, str]]] = {
    "left": [
        ("base", "Panda_leftArm"),
        ("link0", "Panda_leftArm_link0_visual"),
        ("link1", "Panda_leftArm_link1_visual"),
        ("link2", "Panda_leftArm_link2_visual"),
        ("link3", "Panda_leftArm_link3_visual"),
        ("link4", "Panda_leftArm_link4_visual"),
        ("link5", "Panda_leftArm_link5_visual"),
        ("link6", "Panda_leftArm_link6_visual"),
        ("link7", "Panda_leftArm_link7_visual"),
        ("gripper", "Panda_leftArm_gripper_visual"),
        ("finger1", "Panda_leftArm_leftfinger_visual"),
        ("finger2", "Panda_leftArm_rightfinger_visual"),
    ],
    "right": [
        ("base", "Panda_rightArm"),
        ("link0", "Panda_rightArm_link0_visual"),
        ("link1", "Panda_rightArm_link1_visual"),
        ("link2", "Panda_rightArm_link2_visual"),
        ("link3", "Panda_rightArm_link3_visual"),
        ("link4", "Panda_rightArm_link4_visual"),
        ("link5", "Panda_rightArm_link5_visual"),
        ("link6", "Panda_rightArm_link6_visual"),
        ("link7", "Panda_rightArm_link7_visual"),
        ("gripper", "Panda_rightArm_gripper_visual"),
        ("finger1", "Panda_rightArm_leftfinger_visual"),
        ("finger2", "Panda_rightArm_rightfinger_visual"),
    ],
}

def dual_panda_topology_pairs() -> List[Tuple[int, int]]:
    """Directed edges: kinematic chain + gripper→fingers + finger↔finger."""
    undirected: set[Tuple[int, int]] = set()
    # base(0) → link0(1) → … → link7(8) → gripper(9)
    for i in range(9):
        undirected.add((i, i + 1))
    undirected.add((9, 10))  # gripper → finger1
    undirected.add((9, 11))  # gripper → finger2
    undirected.add((10, 11))  # finger1 ↔ finger2

    pairs: List[Tuple[int, int]] = []
    for a, b in sorted(undirected):
        pairs.append((a, b))
        pairs.append((b, a))
    return pairs

TOPOLOGY_PAIRS = dual_panda_topology_pairs()

def _invert_se3(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out

def _transform_points(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    ones = np.ones((pts.shape[0], 1), dtype=np.float64)
    return (T @ np.concatenate([pts, ones], axis=1).T).T[:, :3]

def _sample_shape_local_points(shape, max_points: int, rng: np.random.Generator) -> np.ndarray:
    verts, _, _ = shape.get_mesh_data()
    verts = np.asarray(verts, dtype=np.float64)
    if verts.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float64)
    if verts.shape[0] > max_points:
        idx = rng.choice(verts.shape[0], max_points, replace=False)
        verts = verts[idx]
    return verts

def apply_demo_to_dual_panda(demo_obs, left, right) -> None:
    left.set_joint_positions(
        list(np.asarray(demo_obs.left.joint_positions, dtype=float)),
        disable_dynamics=True,
    )
    right.set_joint_positions(
        list(np.asarray(demo_obs.right.joint_positions, dtype=float)),
        disable_dynamics=True,
    )
    try:
        from pyrep.robots.end_effectors.dual_panda_gripper import (
            PandaGripperLeft,
            PandaGripperRight,
        )

        for g, side in ((PandaGripperLeft(), demo_obs.left), (PandaGripperRight(), demo_obs.right)):
            if hasattr(side, "gripper_joint_positions"):
                g.set_joint_positions(
                    list(np.asarray(side.gripper_joint_positions, dtype=float)),
                    disable_dynamics=True,
                )
            elif hasattr(side, "gripper_open"):
                amount = float(np.asarray(side.gripper_open).reshape(-1)[0])
                _, intervals = g.get_joint_intervals()
                intervals = np.asarray(intervals, dtype=float)
                targets = intervals[:, 0] + (intervals[:, 1] - intervals[:, 0]) * amount
                g.set_joint_positions(list(targets), disable_dynamics=True)
    except Exception:
        pass
    left.set_joint_positions(
        list(np.asarray(demo_obs.left.joint_positions, dtype=float)),
        disable_dynamics=True,
    )
    right.set_joint_positions(
        list(np.asarray(demo_obs.right.joint_positions, dtype=float)),
        disable_dynamics=True,
    )

def sample_arm_from_shapes(
    shapes: Sequence,
    points_per_link: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], List[np.ndarray]]:
    """Returns geometry [N,3], link_means [L,3], local_pts per link, world_T per link."""
    local_pts: List[np.ndarray] = []
    world_T: List[np.ndarray] = []
    world_pts: List[np.ndarray] = []

    for sh in shapes:
        pts_local = _sample_shape_local_points(sh, points_per_link, rng)
        T = np.asarray(sh.get_matrix(), dtype=np.float64).reshape(4, 4)
        if pts_local.shape[0] == 0:
            pts_world = np.zeros((0, 3), dtype=np.float64)
        else:
            pts_world = _transform_points(T, pts_local)
        local_pts.append(pts_local)
        world_T.append(T)
        world_pts.append(pts_world)

    geometry = np.concatenate(world_pts, axis=0).astype(np.float32)
    link_means = np.stack([p.mean(axis=0) if len(p) else np.zeros(3) for p in world_pts], axis=0).astype(
        np.float32
    )
    return geometry, link_means, local_pts, world_T

def build_topology_pc(
    local_pts: List[np.ndarray],
    world_T: List[np.ndarray],
    points_per_link: int,
) -> np.ndarray:
    """[K, M, 5] directed topology edges."""
    rels: List[np.ndarray] = []
    for self_id, nei_id in TOPOLOGY_PAIRS:
        pts_local = local_pts[self_id]
        if pts_local.shape[0] == 0:
            pts_local = np.zeros((points_per_link, 3), dtype=np.float64)
        elif pts_local.shape[0] < points_per_link:
            pad = np.tile(pts_local[-1:], (points_per_link - pts_local.shape[0], 1))
            pts_local = np.concatenate([pts_local, pad], axis=0)
        elif pts_local.shape[0] > points_per_link:
            pts_local = pts_local[:points_per_link]

        T_i = world_T[self_id]
        T_j = world_T[nei_id]
        p_world = _transform_points(T_i, pts_local)
        T_j_inv = _invert_se3(T_j)
        p_in_j = _transform_points(T_j_inv, p_world)
        M = p_in_j.shape[0]
        self_col = np.full((M, 1), float(self_id), dtype=np.float32)
        nei_col = np.full((M, 1), float(nei_id), dtype=np.float32)
        rels.append(np.concatenate([p_in_j.astype(np.float32), self_col, nei_col], axis=1))
    return np.stack(rels, axis=0)

def _sample_dual_panda_robot_pc_from_shapes(
    points_per_link: int,
    seed: int,
) -> Dict[str, np.ndarray]:
    """Sample full-link robot PC from current PyRep shape poses (live scene)."""
    from pyrep.objects.shape import Shape

    rng = np.random.default_rng(seed)
    out: Dict[str, np.ndarray] = {}
    for arm, spec in ARM_SHAPES.items():
        shapes = []
        for _, name in spec:
            try:
                shapes.append(Shape(name))
            except Exception:
                shapes.append(None)
        if any(s is None for s in shapes):
            missing = [n for (_, n), s in zip(spec, shapes) if s is None]
            raise RuntimeError(f"Missing DualPanda shapes for {arm}: {missing}")

        geo, link_means, local_pts, world_T = sample_arm_from_shapes(
            shapes, points_per_link, rng
        )
        topo = build_topology_pc(local_pts, world_T, points_per_link)
        out[f"{arm}_geometry_pc"] = geo
        out[f"{arm}_topology_pc"] = topo
        out[f"{arm}_link_means"] = link_means
    return out

def sample_dual_panda_robot_pc_live(
    points_per_link: int = 16,
    seed: int = 0,
) -> Dict[str, np.ndarray]:
    """Sample full-link robot PC from the running RLBench/PyRep scene (eval rollout)."""
    return _sample_dual_panda_robot_pc_from_shapes(points_per_link, seed)

def sample_dual_panda_robot_pc(
    demo_obs,
    points_per_link: int = 16,
    seed: int = 0,
) -> Dict[str, np.ndarray]:
    """Sample full-link robot PC for one demo timestep (PyRep must be running)."""
    from pyrep.robots.arms.dual_panda import PandaLeft, PandaRight

    apply_demo_to_dual_panda(demo_obs, PandaLeft(), PandaRight())
    return _sample_dual_panda_robot_pc_from_shapes(points_per_link, seed)

ROBOT_PC_NPZ_KEYS = (
    "left_geometry_pc",
    "right_geometry_pc",
    "left_topology_pc",
    "right_topology_pc",
    "left_link_means",
    "right_link_means",
)
