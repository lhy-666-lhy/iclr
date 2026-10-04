#!/usr/bin/env python3
"""Visualize precomputed robot_pc npz (training input) + verify dataloader loads same arrays."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PPI_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PPI_ROOT))

from space.utils.dual_panda_robot_pc import DUAL_PANDA_LINK_NAMES, N_LINKS_PER_ARM

OUT_DIR = PPI_ROOT / "Logs" / "viz_robot_pc"
LINK_COLORS = {
    "base": (0.62, 0.62, 0.62),
    "link0": (0.70, 0.70, 0.70),
    "link1": (0.90, 0.10, 0.30),
    "link2": (0.24, 0.71, 0.29),
    "link3": (1.0, 0.88, 0.10),
    "link4": (0.0, 0.51, 0.78),
    "link5": (0.96, 0.51, 0.19),
    "link6": (0.57, 0.12, 0.71),
    "link7": (0.27, 0.94, 0.94),
    "gripper": (1.0, 0.71, 0.47),
    "finger1": (0.94, 0.20, 0.90),
    "finger2": (0.82, 0.96, 0.24),
}

def split_geometry_by_link(geo: np.ndarray, points_per_link: int) -> dict[str, np.ndarray]:
    parts = {}
    for i, name in enumerate(DUAL_PANDA_LINK_NAMES):
        s, e = i * points_per_link, (i + 1) * points_per_link
        parts[name] = geo[s:e]
    return parts

def load_scene(task: str, episode: int, step: int) -> tuple[np.ndarray, np.ndarray]:
    p = (
        PPI_ROOT
        / "data/training_processed/point_cloud"
        / task
        / "all_variations/episodes"
        / f"episode{episode}/rgb_pcd_rps6144/step{step:03d}.npy"
    )
    arr = np.load(p)
    xyz = arr[:, :3].astype(np.float64)
    rgb = arr[:, 3:6].astype(np.float64)
    if rgb.max() > 1.5:
        rgb /= 255.0
    return xyz, rgb

def load_precomputed_robot_pc(task: str, episode: int, step: int) -> dict[str, np.ndarray]:
    p = (
        PPI_ROOT
        / "data/training_processed/robot_pc"
        / task
        / "all_variations/episodes"
        / f"episode{episode}/step{step:03d}.npz"
    )
    z = np.load(p)
    return {k: z[k] for k in z.files}

def load_via_dataloader(task: str, episode: int, step: int, points_per_link: int) -> dict[str, np.ndarray] | None:
    """Load the same frame through RLBench2Dataset sampler (training path)."""
    try:
        from space.dataset.rlbench2_dataset import RLBench2Dataset
    except Exception as e:
        print(f"[warn] dataloader import failed: {e}")
        return None

    ds = RLBench2Dataset(
        data_path=f"data/training_raw/{task}/all_variations/episodes",
        pcd_path=f"data/training_processed/point_cloud/{task}/all_variations/episodes",
        dino_path=f"data/training_processed/dino_feature/{task}/all_variations/episodes",
        lang_emb_path="data/training_processed/instruction_embeddings.pkl",
        stats_filepath=(
            f"data/training_processed/norm_stats/norm_stats_{task}_"
            "rgb_pcd_rps6144_keyframe_continuous_world_ordered_rps200.pth"
        ),
        point_flow_path=f"data/training_processed/point_flow/{task}/all_variations/episodes",
        robot_pc_path=f"data/training_processed/robot_pc/{task}/all_variations/episodes",
        horizon_keyframe=4,
        horizon_continuous=50,
        start=episode,
        end=episode,
        skip_ep=[],
        pcd_type="rgb_pcd_rps6144",
        prediction_type="keyframe_continuous",
        point_flow_type="world_ordered_rps200",
        task_name="ball_rp1",
        val_ratio=0.0,
        max_train_episodes=1,
    )
    # find a sampler index whose obs step matches
    for idx in range(min(len(ds), 5000)):
        sample = ds.sampler.sample_sequence(idx)
        ep, st = ds.replay_buffer["point_cloud"][ds.sampler.indices[idx][0]]
        if ep == episode and st == step:
            keys = (
                "left_geometry_pc",
                "right_geometry_pc",
                "left_topology_pc",
                "right_topology_pc",
                "left_link_means",
                "right_link_means",
            )
            return {k: sample[k] for k in keys}
    print(f"[warn] no dataloader sample for ep{episode} step{step}")
    return None

def verify_arrays(pre: dict, dl: dict | None) -> str:
    if dl is None:
        return "dataloader check skipped"
    lines = []
    all_ok = True
    for k in sorted(pre.keys()):
        a, b = pre[k], dl[k]
        ok = np.allclose(a, b, rtol=0, atol=0)
        all_ok &= ok
        lines.append(f"  {k}: pre={a.shape} dl={b.shape} match={ok}")
    return ("ALL MATCH ✓" if all_ok else "MISMATCH ✗") + "\n" + "\n".join(lines)

def render_frame(
    task: str,
    episode: int,
    step: int,
    points_per_link: int,
    dl: dict | None,
) -> Path:
    scene_xyz, scene_rgb = load_scene(task, episode, step)
    pre = load_precomputed_robot_pc(task, episode, step)

    idx = np.random.RandomState(0).choice(len(scene_xyz), min(3500, len(scene_xyz)), replace=False)
    sc, sr = scene_xyz[idx], scene_rgb[idx]

    fig = plt.figure(figsize=(16, 6))

    # Panel 1: precomputed robot only (colored by link)
    ax1 = fig.add_subplot(131, projection="3d")
    for arm, prefix in (("left", "L"), ("right", "R")):
        parts = split_geometry_by_link(pre[f"{arm}_geometry_pc"], points_per_link)
        for name, pts in parts.items():
            c = np.array(LINK_COLORS[name])
            if arm == "right":
                c = np.clip(c * np.array([0.85, 0.95, 1.15]), 0, 1)
            ax1.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=[c], s=8, alpha=0.95, label=f"{prefix}_{name}" if step == 0 else None)
    ax1.set_title(f"precomputed robot_pc\nep{episode} step{step} | {N_LINKS_PER_ARM} links × {points_per_link} pts")
    ax1.view_init(22, -60)

    # Panel 2: scene + robot overlay
    ax2 = fig.add_subplot(132, projection="3d")
    ax2.scatter(sc[:, 0], sc[:, 1], sc[:, 2], c=sr, s=1, alpha=0.25)
    finger_pts = []
    finger_cols = []
    for arm in ("left", "right"):
        parts = split_geometry_by_link(pre[f"{arm}_geometry_pc"], points_per_link)
        for name in ("gripper", "finger1", "finger2"):
            pts = parts[name]
            finger_pts.append(pts)
            if name == "finger1":
                c = (0.94, 0.20, 0.90)
            elif name == "finger2":
                c = (0.82, 0.96, 0.24)
            else:
                c = (1.0, 0.55, 0.10)
            finger_cols.append(np.tile(c, (len(pts), 1)))
        for name in DUAL_PANDA_LINK_NAMES:
            if name in ("gripper", "finger1", "finger2"):
                continue
            pts = parts[name]
            ax2.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=[LINK_COLORS[name]], s=3, alpha=0.45)
    fin = np.concatenate(finger_pts, 0)
    fc = np.concatenate(finger_cols, 0)
    ax2.scatter(fin[:, 0], fin[:, 1], fin[:, 2], c=fc, s=14, alpha=1.0)
    ax2.set_title("scene + robot_pc (fingers highlighted)")
    ax2.view_init(22, -60)

    # Panel 3: dataloader vs precomputed diff (if available)
    ax3 = fig.add_subplot(133, projection="3d")
    title3 = "dataloader ≡ precomputed"
    if dl is not None:
        dl_geo = np.concatenate([dl["left_geometry_pc"], dl["right_geometry_pc"]], 0)
        pre_geo = np.concatenate([pre["left_geometry_pc"], pre["right_geometry_pc"]], 0)
        diff = np.linalg.norm(dl_geo - pre_geo, axis=1)
        max_diff = float(diff.max())
        ax3.scatter(pre_geo[:, 0], pre_geo[:, 1], pre_geo[:, 2], c=diff, cmap="coolwarm", s=10, vmin=0, vmax=max(1e-6, max_diff))
        title3 = f"dataloader vs npz | max L2 diff={max_diff:.2e}"
        if max_diff > 0:
            title3 += " MISMATCH"
        else:
            title3 += " ✓ identical"
    else:
        ax3.text2D(0.1, 0.5, "dataloader check\nskipped", transform=ax3.transAxes)
    ax3.set_title(title3)
    ax3.view_init(22, -60)

    for ax in (ax1, ax2, ax3):
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

    tag = f"precomputed_robot_pc_{task}_ep{episode}_step{step:03d}"
    out = OUT_DIR / f"{tag}.png"
    fig.suptitle(f"Training input robot_pc verification | {tag}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return out

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="bimanual_lift_ball")
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--steps", type=int, nargs="+", default=[0, 80])
    p.add_argument("--points_per_link", type=int, default=16)
    args = p.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for step in args.steps:
        pre = load_precomputed_robot_pc(args.task, args.episode, step)
        dl = load_via_dataloader(args.task, args.episode, step, args.points_per_link)
        print(f"\n=== ep{args.episode} step{step} ===")
        print(f"left_geometry_pc {pre['left_geometry_pc'].shape}  link_means {pre['left_link_means'].shape}")
        print(f"left_topology_pc {pre['left_topology_pc'].shape}")
        print(verify_arrays(pre, dl))
        out = render_frame(args.task, args.episode, step, args.points_per_link, dl)
        print(f"[ok] wrote {out}")

if __name__ == "__main__":
    main()
