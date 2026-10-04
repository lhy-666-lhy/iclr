#!/usr/bin/env python3
"""Emit (task, episode) jobs for parallel robot_pc precompute. Skips fully-done episodes."""

from __future__ import annotations

import argparse
from pathlib import Path

PPI_ROOT = Path(__file__).resolve().parents[2]

TASKS = [
    "bimanual_handover_item_easy",
    "bimanual_lift_ball",
    "bimanual_lift_tray",
    "bimanual_pick_laptop",
    "bimanual_push_box",
    "bimanual_put_item_in_drawer",
    "bimanual_sweep_to_dustpan",
]

def episode_todo(task: str, ep: int, overwrite: bool) -> int:
    pcd_dir = (
        PPI_ROOT
        / "data/training_processed/point_cloud"
        / task
        / "all_variations/episodes"
        / f"episode{ep}/rgb_pcd_rps6144"
    )
    out_dir = (
        PPI_ROOT
        / "data/training_processed/robot_pc"
        / task
        / "all_variations/episodes"
        / f"episode{ep}"
    )
    if not pcd_dir.exists():
        return 0
    steps = sorted(int(p.stem.replace("step", "")) for p in pcd_dir.glob("step*.npy"))
    if overwrite:
        return len(steps)
    return sum(1 for s in steps if not (out_dir / f"step{s:03d}.npz").exists())

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=104)
    p.add_argument("--tasks", nargs="*", default=TASKS)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--count_only", action="store_true")
    args = p.parse_args()

    jobs = []
    for task in args.tasks:
        for ep in range(args.start, args.end + 1):
            n = episode_todo(task, ep, args.overwrite)
            if n > 0:
                jobs.append((task, ep, n))

    if args.count_only:
        total_steps = sum(j[2] for j in jobs)
        print(f"jobs={len(jobs)} pending_steps={total_steps}")
        for task in args.tasks:
            tjobs = [j for j in jobs if j[0] == task]
            print(f"  {task}: {len(tjobs)} episodes, {sum(j[2] for j in tjobs)} steps")
        return

    for task, ep, _ in jobs:
        print(f"{task}\t{ep}")

if __name__ == "__main__":
    main()
