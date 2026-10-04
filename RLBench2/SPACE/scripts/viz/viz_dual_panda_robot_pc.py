#!/usr/bin/env python3
"""Visualize DualPanda link meshes (from CoppeliaSim/PyRep) over obs point cloud.

Uses the real DualPanda scene kinematics (correct root & orientation), NOT
Franka-URDF EE-alignment (which misplaces the arm body).

Example:
  COPPELIASIM_ROOT=.../CoppeliaSim \\
  python scripts/viz/viz_dual_panda_robot_pc.py --episode 0 --step 80
"""

from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

PPI_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PPI_ROOT / "Logs" / "viz_robot_pc"
DEFAULT_SCENE = Path(
    os.environ.get("COPPELIASIM_ROOT", "")
    "rlbench/task_design_bimanual.ttt"
)

# Per-link colors (RGB 0-255). Keys = short link id.
LINK_COLORS = {
    "base": (160, 160, 160),
    "link0": (180, 180, 180),
    "link1": (230, 25, 75),
    "link2": (60, 180, 75),
    "link3": (255, 225, 25),
    "link4": (0, 130, 200),
    "link5": (245, 130, 48),
    "link6": (145, 30, 180),
    "link7": (70, 240, 240),
    "gripper": (255, 180, 120),
    "finger1": (240, 50, 230),
    "finger2": (210, 245, 60),
}

# Shape names in DualPanda Coppelia model (arm + gripper body + two fingers)
ARM_SHAPES = {
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

def load_obs_pc(pc_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.load(pc_path)
    xyz = arr[:, :3].astype(np.float64)
    if arr.shape[1] >= 6:
        rgb = arr[:, 3:6].astype(np.float64)
        if rgb.max() > 1.5:
            rgb = rgb / 255.0
    else:
        rgb = np.full((xyz.shape[0], 3), 0.55)
    return xyz, rgb

def _shape_world_points(shape, max_points: int, rng: np.random.Generator) -> np.ndarray:
    """Local mesh verts → world via shape.get_matrix()."""
    verts, _, _ = shape.get_mesh_data()
    verts = np.asarray(verts, dtype=np.float64)
    if verts.shape[0] == 0:
        return np.zeros((0, 3))
    if verts.shape[0] > max_points:
        idx = rng.choice(verts.shape[0], max_points, replace=False)
        verts = verts[idx]
    T = np.asarray(shape.get_matrix(), dtype=np.float64).reshape(4, 4)
    ones = np.ones((verts.shape[0], 1), dtype=np.float64)
    return (T @ np.concatenate([verts, ones], axis=1).T).T[:, :3]

def sample_dual_panda_clouds(
    demo_obs,
    scene_ttt: Path,
    points_per_link: int = 400,
) -> Dict[str, Dict[str, np.ndarray]]:
    from pyrep import PyRep
    from pyrep.objects.shape import Shape
    from pyrep.robots.arms.dual_panda import PandaLeft, PandaRight

    pr = PyRep()
    pr.launch(str(scene_ttt), headless=True)
    pr.start()
    try:
        left = PandaLeft()
        right = PandaRight()
        left.set_joint_positions(
            list(np.asarray(demo_obs.left.joint_positions, dtype=float)),
            disable_dynamics=True,
        )
        right.set_joint_positions(
            list(np.asarray(demo_obs.right.joint_positions, dtype=float)),
            disable_dynamics=True,
        )
        # set gripper finger openings to match demo
        try:
            from pyrep.robots.end_effectors.dual_panda_gripper import (
                PandaGripperLeft,
                PandaGripperRight,
            )
            g_l, g_r = PandaGripperLeft(), PandaGripperRight()
            for g, side in ((g_l, demo_obs.left), (g_r, demo_obs.right)):
                if hasattr(side, "gripper_joint_positions"):
                    g.set_joint_positions(
                        list(np.asarray(side.gripper_joint_positions, dtype=float)),
                        disable_dynamics=True,
                    )
                elif hasattr(side, "gripper_open"):
                    # 1=open, 0=closed → map via joint intervals
                    amount = float(np.asarray(side.gripper_open).reshape(-1)[0])
                    _, intervals = g.get_joint_intervals()
                    intervals = np.asarray(intervals, dtype=float)
                    targets = intervals[:, 0] + (intervals[:, 1] - intervals[:, 0]) * amount
                    g.set_joint_positions(list(targets), disable_dynamics=True)
                print(
                    f"[check] {g.get_name()} joints={g.get_joint_positions()} "
                    f"open≈{getattr(side, 'gripper_open', None)}"
                )
        except Exception as e:
            print(f"[warn] gripper set failed: {e}")
        # re-apply arm joints after gripper (some Coppelia models nudge parent links)
        left.set_joint_positions(
            list(np.asarray(demo_obs.left.joint_positions, dtype=float)),
            disable_dynamics=True,
        )
        right.set_joint_positions(
            list(np.asarray(demo_obs.right.joint_positions, dtype=float)),
            disable_dynamics=True,
        )
        pr.step()

        tip_l = np.asarray(left.get_tip().get_pose(), dtype=np.float64)
        tip_r = np.asarray(right.get_tip().get_pose(), dtype=np.float64)
        gp_l = np.asarray(demo_obs.left.gripper_pose, dtype=np.float64)
        gp_r = np.asarray(demo_obs.right.gripper_pose, dtype=np.float64)
        print(f"[check] left tip vs demo |d|={np.linalg.norm(tip_l[:3]-gp_l[:3]):.4f} m")
        print(f"[check] right tip vs demo |d|={np.linalg.norm(tip_r[:3]-gp_r[:3]):.4f} m")
        print(f"[check] left base pose={np.asarray(Shape('Panda_leftArm').get_pose())}")
        print(f"[check] right base pose={np.asarray(Shape('Panda_rightArm').get_pose())}")

        rng = np.random.default_rng(0)
        out: Dict[str, Dict[str, np.ndarray]] = {"left": {}, "right": {}}
        for arm, shapes in ARM_SHAPES.items():
            for short, name in shapes:
                try:
                    sh = Shape(name)
                except Exception as e:
                    print(f"[warn] missing {name}: {e}")
                    continue
                pts = _shape_world_points(sh, points_per_link, rng)
                out[arm][short] = pts
                print(f"[info] {arm}/{short}: n={pts.shape[0]} mean={pts.mean(0)}")
        return out
    finally:
        pr.stop()
        pr.shutdown()

def build_plotly_figure(scene_xyz, scene_rgb, arm_clouds, title: str, tips=None):
    import plotly.graph_objects as go

    fig = go.Figure()
    n = scene_xyz.shape[0]
    idx = np.arange(n) if n <= 8000 else np.random.default_rng(0).choice(n, 8000, replace=False)
    sx, sc = scene_xyz[idx], scene_rgb[idx]
    colors = [f"rgb({int(r*255)},{int(g*255)},{int(b*255)})" for r, g, b in sc]
    fig.add_trace(
        go.Scatter3d(
            x=sx[:, 0], y=sx[:, 1], z=sx[:, 2],
            mode="markers",
            marker=dict(size=2.0, color=colors, opacity=0.55),
            name="obs_scene",
        )
    )
    for arm, links in arm_clouds.items():
        for link, pts in links.items():
            if pts.size == 0:
                continue
            c = LINK_COLORS.get(link, (200, 200, 200))
            if arm == "right":
                c = tuple(int(np.clip(v * t, 0, 255)) for v, t in zip(c, (0.85, 0.95, 1.15)))
            col = f"rgb({c[0]},{c[1]},{c[2]})"
            fig.add_trace(
                go.Scatter3d(
                    x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                    mode="markers",
                    marker=dict(size=2.4, color=col, opacity=0.9),
                    name=f"{arm}/{link}",
                )
            )
    if tips:
        for name, p in tips.items():
            fig.add_trace(
                go.Scatter3d(
                    x=[p[0]], y=[p[1]], z=[p[2]],
                    mode="markers",
                    marker=dict(size=8, color="black", symbol="x"),
                    name=name,
                )
            )
    fig.update_layout(
        title=title,
        scene=dict(aspectmode="data", xaxis_title="x", yaxis_title="y", zaxis_title="z"),
        legend=dict(itemsizing="constant"),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig

def save_matplotlib_preview(scene_xyz, scene_rgb, arm_clouds, out_png: Path, title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    idx = np.random.default_rng(0).choice(len(scene_xyz), min(4000, len(scene_xyz)), replace=False)
    ax.scatter(
        scene_xyz[idx, 0], scene_xyz[idx, 1], scene_xyz[idx, 2],
        c=scene_rgb[idx], s=1, alpha=0.25,
    )
    for arm, links in arm_clouds.items():
        for link, pts in links.items():
            if pts.size == 0:
                continue
            c = np.array(LINK_COLORS.get(link, (200, 200, 200)), dtype=np.float64) / 255.0
            if arm == "right":
                c = np.clip(c * np.array([0.85, 0.95, 1.15]), 0, 1)
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=4, c=[c], alpha=0.95)
    ax.set_title(title)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    mid = scene_xyz.mean(0)
    r = np.percentile(np.linalg.norm(scene_xyz - mid, axis=1), 95)
    ax.set_xlim(mid[0] - r, mid[0] + r)
    ax.set_ylim(mid[1] - r, mid[1] + r)
    ax.set_zlim(mid[2] - r, mid[2] + r)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="bimanual_lift_ball")
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--step", type=int, default=80)
    ap.add_argument("--points-per-link", type=int, default=400)
    ap.add_argument("--scene-ttt", type=Path, default=DEFAULT_SCENE)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    if "COPPELIASIM_ROOT" not in os.environ:
        raise RuntimeError("Set COPPELIASIM_ROOT before running this script")

    raw_ep = (
        PPI_ROOT / "data/training_raw" / args.task / "all_variations/episodes" / f"episode{args.episode}"
    )
    pc_path = (
        PPI_ROOT
        / "data/training_processed/point_cloud"
        / args.task
        / "all_variations/episodes"
        / f"episode{args.episode}"
        / "rgb_pcd_rps6144"
        / f"step{args.step:03d}.npy"
    )
    with open(raw_ep / "low_dim_obs.pkl", "rb") as f:
        demo = pickle.load(f)
    obs = demo[args.step]
    scene_xyz, scene_rgb = load_obs_pc(pc_path)

    print(f"[info] scene={args.scene_ttt}")
    print(f"[info] obs_pc={pc_path}")
    arm_clouds = sample_dual_panda_clouds(obs, args.scene_ttt, args.points_per_link)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.task}_ep{args.episode}_step{args.step:03d}"
    html_path = args.out_dir / f"{stem}_dual_panda_overlay.html"
    npz_path = args.out_dir / f"{stem}_robot_links.npz"
    png_path = args.out_dir / f"{stem}_preview.png"

    save = {"scene_xyz": scene_xyz, "scene_rgb": scene_rgb}
    for arm, links in arm_clouds.items():
        for link, pts in links.items():
            save[f"{arm}__{link}"] = pts
    np.savez_compressed(npz_path, **save)

    tips = {
        "demo_left_tip": np.asarray(obs.left.gripper_pose[:3], dtype=float),
        "demo_right_tip": np.asarray(obs.right.gripper_pose[:3], dtype=float),
    }
    title = f"DualPanda (PyRep FK) + obs PC | {args.task} ep{args.episode} step{args.step}"
    fig = build_plotly_figure(scene_xyz, scene_rgb, arm_clouds, title, tips=tips)
    fig.write_html(str(html_path), include_plotlyjs="cdn")
    save_matplotlib_preview(scene_xyz, scene_rgb, arm_clouds, png_path, title)
    print(f"[ok] html → {html_path}")
    print(f"[ok] png  → {png_path}")
    print(f"[ok] npz  → {npz_path}")

if __name__ == "__main__":
    main()
