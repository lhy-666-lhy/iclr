#!/usr/bin/env python3
"""Precompute DualPanda full-link robot PCs for PPI_rp1 training.

Output layout:
  data/training_processed/robot_pc/<task>/all_variations/episodes/
    episode{e}/step{t:03d}.npz

Each npz contains:
  left/right_geometry_pc, left/right_topology_pc, left/right_link_means

Usage:
  export COPPELIASIM_ROOT=.../CoppeliaSim
  export LD_LIBRARY_PATH=$COPPELIASIM_ROOT:$LD_LIBRARY_PATH
  export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
  export QT_QPA_PLATFORM=offscreen
  python scripts/data_generation/precompute_dual_panda_robot_pc.py \\
    --task bimanual_lift_ball --start 0 --end 99 --points_per_link 16
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

PPI_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PPI_ROOT))

from space.utils.dual_panda_robot_pc import ROBOT_PC_NPZ_KEYS, sample_dual_panda_robot_pc

DEFAULT_SCENE = Path(
    os.environ.get("COPPELIASIM_ROOT", "")
    "rlbench/task_design_bimanual.ttt"
)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="bimanual_lift_ball")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=99)
    p.add_argument("--points_per_link", type=int, default=16)
    p.add_argument("--scene_ttt", type=Path, default=DEFAULT_SCENE)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()

def main():
    args = parse_args()
    raw_root = PPI_ROOT / "data/training_raw" / args.task / "all_variations/episodes"
    out_root = PPI_ROOT / "data/training_processed/robot_pc" / args.task / "all_variations/episodes"
    pcd_root = PPI_ROOT / "data/training_processed/point_cloud" / args.task / "all_variations/episodes"

    if not raw_root.exists():
        raise FileNotFoundError(f"raw data not found: {raw_root}")
    if not args.scene_ttt.exists():
        raise FileNotFoundError(f"scene not found: {args.scene_ttt}")

    episodes = list(range(args.start, args.end + 1))
    todo = []
    for ep in episodes:
        ep_dir = pcd_root / f"episode{ep}" / "rgb_pcd_rps6144"
        if not ep_dir.exists():
            print(f"[skip] episode{ep}: no scene pcd dir")
            continue
        steps = sorted(int(p.stem.replace("step", "")) for p in ep_dir.glob("step*.npy"))
        for step in steps:
            out_path = out_root / f"episode{ep}" / f"step{step:03d}.npz"
            if out_path.exists() and not args.overwrite:
                continue
            todo.append((ep, step, out_path))

    print(f"[INFO] task={args.task} episodes={args.start}-{args.end} todo={len(todo)} M={args.points_per_link}")
    if args.dry_run:
        for item in todo[:20]:
            print(" ", item[2])
        if len(todo) > 20:
            print(f"  ... +{len(todo)-20} more")
        return

    from pyrep import PyRep

    pr = PyRep()
    pr.launch(str(args.scene_ttt), headless=True)
    pr.start()
    done = 0
    try:
        for ep, step, out_path in todo:
            low_dim = raw_root / f"episode{ep}" / "low_dim_obs.pkl"
            if not low_dim.exists():
                print(f"[warn] missing {low_dim}")
                continue
            with open(low_dim, "rb") as f:
                obs_list = pickle.load(f)
            if step >= len(obs_list):
                print(f"[warn] ep{ep} step{step} out of range ({len(obs_list)})")
                continue

            demo_obs = obs_list[step]
            pr.step()
            pc = sample_dual_panda_robot_pc(
                demo_obs, points_per_link=args.points_per_link, seed=ep * 10000 + step
            )
            out_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(out_path, **pc)
            done += 1
            if done % 50 == 0 or done == len(todo):
                print(f"[{done}/{len(todo)}] saved {out_path.name} ep{ep} geo={pc['left_geometry_pc'].shape}")
    finally:
        pr.stop()
        pr.shutdown()

    print(f"[DONE] wrote {done} npz files under {out_root}")

if __name__ == "__main__":
    main()
