#!/usr/bin/env python3
"""Export paper-ready PNG figures for robot mesh / geometry / topology PCs.

Figures (left arm, ab_exp2-style surface sampling):
  01  Collision mesh of a mid-arm link + a gripper finger
  02  Per-link local surface point clouds (color = link id)
  03  Geometry PC in world frame (all links, color = link id)
  04  Topology PC panels (self points in neighbor frame)
  05  Neighbors in one link frame + this link mesh at origin

Usage:
  python policy/SPACE/scripts/export_paper_robot_pc_figs.py \\
      --out_dir policy/SPACE/docs/figures/robot_pc_paper \\
      --points_per_link 16
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

BLOCK_DIR = Path(__file__).resolve().parents[1] / "block"
if str(BLOCK_DIR) not in sys.path:
    sys.path.insert(0, str(BLOCK_DIR))

from make_robot import (  # noqa: E402
    RobotPointCloudGenerator,
    _discover_arm_links,
    _invert_se3,
    _preferred_arm_link_order,
    _transform_local_points,
)

# Paper-friendly distinct colors (link0..link8)
LINK_COLORS = np.array(
    [
        [0.839, 0.153, 0.157],  # base  red
        [1.000, 0.498, 0.055],  # 1     orange
        [0.122, 0.467, 0.706],  # 2     blue
        [0.173, 0.627, 0.173],  # 3     green  (mid)
        [0.580, 0.404, 0.741],  # 4     purple
        [0.549, 0.337, 0.294],  # 5     brown
        [0.890, 0.467, 0.761],  # 6     pink   (wrist)
        [0.498, 0.498, 0.498],  # 7     gray   (finger)
        [0.737, 0.741, 0.133],  # 8     olive  (finger)
    ],
    dtype=np.float64,
)

def _short(name: str) -> str:
    return name.split("_", 1)[-1] if "_" in name else name

def gen_idx(name: str) -> int:
    if "base" in name:
        return 0
    for i in range(1, 9):
        if name.endswith(f"link{i}") or name.split("_")[-1] == f"link{i}":
            return i
    return 0

def _set_equal_aspect(ax, pts: np.ndarray, pad: float = 0.08) -> None:
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
    if pts.size == 0:
        return
    c = pts.mean(axis=0)
    r = np.max(np.linalg.norm(pts - c, axis=1))
    r = max(float(r) * (1.0 + pad), 1e-3)
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(c[2] - r, c[2] + r)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass

def _style_ax(ax, title: str = "") -> None:
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    if title:
        ax.set_title(title, fontsize=11, pad=8)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(True, alpha=0.25)

def _plot_mesh(
    ax,
    mesh,
    color=(0.45, 0.55, 0.75),
    alpha: float = 0.35,
    edge: bool = True,
    max_faces: int = 12000,
) -> np.ndarray:
    """Draw trimesh on Axes3D; return vertices used for aspect."""
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if len(faces) > max_faces:
        rng = np.random.RandomState(0)
        faces = faces[rng.choice(len(faces), size=max_faces, replace=False)]
    tris = verts[faces]
    coll = Poly3DCollection(
        tris,
        alpha=alpha,
        facecolor=color,
        edgecolor=(0.15, 0.15, 0.15, 0.15) if edge else "none",
        linewidths=0.15,
    )
    ax.add_collection3d(coll)
    return verts

def _scatter(ax, pts: np.ndarray, color, s: float = 8, label: Optional[str] = None, alpha: float = 0.95):
    pts = np.asarray(pts).reshape(-1, 3)
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=[color], s=s, depthshade=True, alpha=alpha, label=label, linewidths=0)

def _save(fig, path: Path, dpi: int = 300) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[saved] {path}")

def _load_left_meshes(gen: RobotPointCloudGenerator) -> Dict[str, object]:
    names, meshes = _discover_arm_links(gen.urdf_path, gen.left_prefix)
    ordered = _preferred_arm_link_order(names, gen.left_prefix)
    mmap = dict(zip(names, meshes))
    return {n: mmap[n] for n in ordered if n in mmap and mmap[n] is not None}

def _per_link_world(
    gen: RobotPointCloudGenerator, qpos: np.ndarray
) -> List[np.ndarray]:
    q = torch.as_tensor(qpos, dtype=torch.float32).unsqueeze(0)
    fk = gen.forward_kinematics(q)
    clouds: List[np.ndarray] = []
    for name, pts_local in zip(gen.left_links, gen.left_local):
        pw = _transform_local_points(fk["left"][name], pts_local)[0]
        clouds.append(pw.detach().cpu().numpy())
    return clouds

def fig_meshes(mesh_map: Dict[str, object], mid_name: str, grip_name: str, out_dir: Path, dpi: int):
    fig = plt.figure(figsize=(10.5, 4.6))
    for i, (name, tag) in enumerate([(mid_name, "mid-arm link"), (grip_name, "gripper finger")]):
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        mesh = mesh_map[name]
        verts = _plot_mesh(ax, mesh, color=LINK_COLORS[gen_idx(name)], alpha=0.42)
        # denser surface samples for visual context
        try:
            import trimesh

            dens, _ = trimesh.sample.sample_surface(mesh, 800, seed=0)
            _scatter(ax, dens, LINK_COLORS[gen_idx(name)], s=2, alpha=0.35)
            all_pts = np.vstack([verts, dens])
        except Exception:
            all_pts = verts
        _set_equal_aspect(ax, all_pts)
        _style_ax(ax, f"{_short(name)}  ({tag})")
        ax.view_init(elev=22, azim=35)
    fig.suptitle("Left-arm collision mesh examples (link-local frame)", fontsize=12, y=1.02)
    _save(fig, out_dir / "01_mesh_mid_and_gripper.png", dpi=dpi)

def fig_local_link_pcs(
    gen: RobotPointCloudGenerator, out_dir: Path, dpi: int, denser_m: int = 64
):
    """Per-link local PC gallery + one combined local-frame strip (translated for display)."""
    n = len(gen.left_links)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig = plt.figure(figsize=(11, 3.2 * nrows))
    for i, (name, pts_t) in enumerate(zip(gen.left_links, gen.left_local)):
        ax = fig.add_subplot(nrows, ncols, i + 1, projection="3d")
        pts = pts_t.detach().cpu().numpy()
        # upsample only for prettier paper dots if M is small
        mesh_pts = pts
        if denser_m > pts.shape[0]:
            # keep exact training samples as large dots; optional denser behind
            pass
        _scatter(ax, mesh_pts, LINK_COLORS[i % len(LINK_COLORS)], s=18, label=_short(name))
        _set_equal_aspect(ax, mesh_pts)
        _style_ax(ax, f"[{i}] {_short(name)}  M={pts.shape[0]}")
        ax.view_init(elev=20, azim=40)
    fig.suptitle("Link-level surface point clouds (each in its own link frame)", fontsize=12)
    fig.tight_layout()
    _save(fig, out_dir / "02_link_local_pointclouds.png", dpi=dpi)

    # Combined: place each local cloud with a lateral offset for a single overview
    fig2 = plt.figure(figsize=(12, 4.5))
    ax = fig2.add_subplot(111, projection="3d")
    all_pts = []
    for i, (name, pts_t) in enumerate(zip(gen.left_links, gen.left_local)):
        pts = pts_t.detach().cpu().numpy().copy()
        pts = pts - pts.mean(axis=0, keepdims=True)
        pts[:, 0] += i * 0.12  # separate along X for readability
        _scatter(ax, pts, LINK_COLORS[i % len(LINK_COLORS)], s=14, label=f"[{i}] {_short(name)}")
        all_pts.append(pts)
    _set_equal_aspect(ax, np.vstack(all_pts), pad=0.12)
    _style_ax(ax, "Link PCs (local, centered; spaced along X for display)")
    ax.legend(loc="upper left", fontsize=7, ncol=3, frameon=False)
    ax.view_init(elev=18, azim=-70)
    fig2.tight_layout()
    _save(fig2, out_dir / "02b_link_local_pointclouds_strip.png", dpi=dpi)

def fig_geometry_world(gen: RobotPointCloudGenerator, qpos: np.ndarray, out_dir: Path, dpi: int):
    clouds = _per_link_world(gen, qpos)
    fig = plt.figure(figsize=(7.2, 6.2))
    ax = fig.add_subplot(111, projection="3d")
    all_pts = []
    for i, (name, pts) in enumerate(zip(gen.left_links, clouds)):
        _scatter(ax, pts, LINK_COLORS[i % len(LINK_COLORS)], s=12, label=f"[{i}] {_short(name)}")
        all_pts.append(pts)
    stack = np.vstack(all_pts)
    _set_equal_aspect(ax, stack)
    _style_ax(ax, "Left-arm geometry PC (world frame, color = link)")
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    ax.view_init(elev=22, azim=50)
    fig.tight_layout()
    _save(fig, out_dir / "03_geometry_pc_world_colored.png", dpi=dpi)

def fig_topology(
    gen: RobotPointCloudGenerator,
    topo: np.ndarray,
    out_dir: Path,
    dpi: int,
    max_panels: int = 9,
):
    """Show directed topology relations: points of self in neighbor frame."""
    pairs = gen.left_neighbor_pairs
    # Prefer mid-chain + gripper edges for paper
    preferred = []
    for k, (i, j) in enumerate(pairs):
        key = (min(i, j), max(i, j))
        if key in {(2, 3), (3, 4), (5, 6), (6, 7), (6, 8), (7, 8)} or i == 3 or j == 3:
            preferred.append(k)
    # keep order unique
    seen = set()
    sel = []
    for k in preferred + list(range(len(pairs))):
        if k not in seen:
            seen.add(k)
            sel.append(k)
        if len(sel) >= max_panels:
            break

    ncols = 3
    nrows = int(np.ceil(len(sel) / ncols))
    fig = plt.figure(figsize=(11.5, 3.4 * nrows))
    for pi, k in enumerate(sel):
        ax = fig.add_subplot(nrows, ncols, pi + 1, projection="3d")
        i, j = pairs[k]
        pts = topo[k, :, :3]
        # color by self link
        _scatter(ax, pts, LINK_COLORS[i % len(LINK_COLORS)], s=16)
        # origin = neighbor frame
        ax.scatter([0], [0], [0], c="k", s=40, marker="x")
        _set_equal_aspect(ax, np.vstack([pts, np.zeros((1, 3))]))
        _style_ax(
            ax,
            f"rel[{k}]  {_short(gen.left_links[i])} → {_short(gen.left_links[j])}\n"
            f"(xyz in neighbor frame; × = neighbor origin)",
        )
        ax.view_init(elev=20, azim=40)
    fig.suptitle(
        "Topology PC: self-link surface points expressed in neighbor link frame",
        fontsize=12,
    )
    fig.tight_layout()
    _save(fig, out_dir / "04_topology_pc_neighbor_frames.png", dpi=dpi)

    # Compact single overview: color by relation index, all in one plot is messy;
    # instead save a gripper-focused trio.
    grip_ks = [k for k, (i, j) in enumerate(pairs) if {i, j} <= {6, 7, 8} or (6 in (i, j) and max(i, j) >= 7)]
    grip_ks = grip_ks[:6]
    if grip_ks:
        fig2 = plt.figure(figsize=(11, 6.5))
        for pi, k in enumerate(grip_ks):
            ax = fig2.add_subplot(2, 3, pi + 1, projection="3d")
            i, j = pairs[k]
            pts = topo[k, :, :3]
            _scatter(ax, pts, LINK_COLORS[i % len(LINK_COLORS)], s=18)
            ax.scatter([0], [0], [0], c="k", s=45, marker="x")
            _set_equal_aspect(ax, np.vstack([pts, np.zeros((1, 3))]))
            _style_ax(ax, f"{_short(gen.left_links[i])} → {_short(gen.left_links[j])}")
            ax.view_init(elev=22, azim=35)
        fig2.suptitle("Gripper topology relations (6↔7, 6↔8, 7↔8)", fontsize=12)
        fig2.tight_layout()
        _save(fig2, out_dir / "04b_topology_gripper_relations.png", dpi=dpi)

def fig_neighbors_in_anchor_frame(
    gen: RobotPointCloudGenerator,
    mesh_map: Dict[str, object],
    qpos: np.ndarray,
    anchor_name: str,
    out_dir: Path,
    dpi: int,
):
    """Anchor link mesh at origin; adjacent links as PCs in that frame."""
    if anchor_name not in gen.left_links:
        raise KeyError(anchor_name)
    a = gen.left_links.index(anchor_name)
    neighbors = sorted({j if i == a else i for i, j in gen.left_neighbor_pairs if a in (i, j)})

    q = torch.as_tensor(qpos, dtype=torch.float32).unsqueeze(0)
    fk = gen.forward_kinematics(q)
    T_a = fk["left"][anchor_name]  # (1,4,4)
    T_a_inv = _invert_se3(T_a)

    fig = plt.figure(figsize=(7.5, 6.5))
    ax = fig.add_subplot(111, projection="3d")

    # Anchor mesh at origin (already in link-local / collision frame)
    mesh = mesh_map[anchor_name]
    verts = _plot_mesh(ax, mesh, color=(*LINK_COLORS[a % len(LINK_COLORS)],), alpha=0.30, edge=True)
    all_pts = [verts]

    # Also show anchor local samples lightly
    pts_a = gen.left_local[a].detach().cpu().numpy()
    _scatter(ax, pts_a, LINK_COLORS[a % len(LINK_COLORS)], s=8, alpha=0.45, label=f"[{a}] {_short(anchor_name)} (self)")

    for nidx in neighbors:
        name = gen.left_links[nidx]
        pts_local = gen.left_local[nidx]
        T_n = fk["left"][name]
        # world then into anchor: inv(T_a) @ T_n @ p_local
        p_world = _transform_local_points(T_n, pts_local)  # (1,M,3)
        ones = torch.ones(1, p_world.shape[1], 1)
        p_h = torch.cat([p_world, ones], dim=-1)
        p_in_a = torch.einsum("bij,bmj->bmi", T_a_inv, p_h)[..., :3][0].detach().cpu().numpy()
        _scatter(
            ax,
            p_in_a,
            LINK_COLORS[nidx % len(LINK_COLORS)],
            s=22,
            label=f"[{nidx}] {_short(name)} (neighbor PC)",
        )
        all_pts.append(p_in_a)

    stack = np.vstack(all_pts)
    _set_equal_aspect(ax, stack, pad=0.15)
    _style_ax(ax, f"Neighbors in {_short(anchor_name)} frame\n(mesh @ origin; colored = adjacent link PCs)")
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    ax.view_init(elev=20, azim=45)
    ax.scatter([0], [0], [0], c="k", s=50, marker="+")
    fig.tight_layout()
    _save(fig, out_dir / f"05_neighbors_in_{_short(anchor_name)}_frame.png", dpi=dpi)

    # Extra: mid-link version if anchor is wrist — also export link3 version
    return neighbors

def build_demo_qpos(seed: int = 0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    q = rng.uniform(-0.35, 0.35, size=14).astype(np.float32)
    # Slightly open gripper for finger separation visibility
    q[6] = 0.55
    q[13] = 0.55
    # Make mid joints a bit more posed for geometry spread
    q[2] = 0.55
    q[3] = -0.40
    q[4] = 0.35
    return q

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out_dir",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "docs" / "figures" / "robot_pc_paper"),
    )
    parser.add_argument("--points_per_link", type=int, default=16, help="Match ab_exp2 M=16 by default")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mid_link", type=str, default="fl_link3")
    parser.add_argument("--grip_link", type=str, default="fl_link7")
    parser.add_argument("--anchor_link", type=str, default="fl_link6", help="Frame origin for neighbor viz")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gen = RobotPointCloudGenerator(
        points_per_link=args.points_per_link,
        sample_region="surface",
        device="cpu",
    )
    mesh_map = _load_left_meshes(gen)
    qpos = build_demo_qpos(args.seed)

    print("Left links:", gen.left_links)
    print("Neighbor pairs (directed):", gen.left_neighbor_pairs)
    print("qpos:", np.round(qpos, 3).tolist())

    # Resolve names if user typos
    for attr in ("mid_link", "grip_link", "anchor_link"):
        name = getattr(args, attr)
        if name not in gen.left_links:
            raise SystemExit(f"{attr}={name!r} not in {gen.left_links}")

    fig_meshes(mesh_map, args.mid_link, args.grip_link, out_dir, args.dpi)
    fig_local_link_pcs(gen, out_dir, args.dpi)
    fig_geometry_world(gen, qpos, out_dir, args.dpi)

    robot = gen.make_robot(torch.from_numpy(qpos))
    topo = robot["left_topology_pc"].detach().cpu().numpy()
    fig_topology(gen, topo, out_dir, args.dpi)
    fig_neighbors_in_anchor_frame(gen, mesh_map, qpos, args.anchor_link, out_dir, args.dpi)

    # Also export mid-link anchor (link3) for the "middle link" story
    if args.anchor_link != args.mid_link:
        fig_neighbors_in_anchor_frame(gen, mesh_map, qpos, args.mid_link, out_dir, args.dpi)

    # README for the figure pack
    readme = out_dir / "README.md"
    readme.write_text(
        f"""# Paper robot PC figures

Generated with `export_paper_robot_pc_figs.py` (surface sampling, M={args.points_per_link}).

| File | Content |
|------|---------|
| `01_mesh_mid_and_gripper.png` | Collision meshes: `{args.mid_link}` + `{args.grip_link}` |
| `02_link_local_pointclouds.png` | Per-link local surface PCs (color by link) |
| `02b_link_local_pointclouds_strip.png` | Same PCs, spaced strip overview |
| `03_geometry_pc_world_colored.png` | Left-arm geometry PC in **world** frame |
| `04_topology_pc_neighbor_frames.png` | Topology: self points in **neighbor** frame |
| `04b_topology_gripper_relations.png` | Gripper-focused topology relations |
| `05_neighbors_in_link6_frame.png` | Mesh @ origin + adjacent link PCs in that frame |
| `05_neighbors_in_link3_frame.png` | Same for mid link (if requested) |

Demo qpos seed={args.seed}.
""",
        encoding="utf-8",
    )
    print(f"Done → {out_dir}")

if __name__ == "__main__":
    main()
