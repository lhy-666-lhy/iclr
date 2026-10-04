#!/usr/bin/env python3
"""Precompute robot geometry/topology point clouds into existing SPACE zarr.

Writes flat keys under data/:
  robot_geo_{left,right}_{region}_m{M}   [T, 9M, 3]
  robot_topo_{left,right}_{region}_m{M}  [T, 18, M, 5]

Usage (from policy/SPACE):
  python scripts/precompute_robot_pc.py handover_block demo_clean 100
  python scripts/precompute_robot_pc.py --all demo_clean 100
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import zarr
from numcodecs import Blosc

BLOCK_DIR = Path(__file__).resolve().parents[1] / "block"
if str(BLOCK_DIR) not in sys.path:
    sys.path.insert(0, str(BLOCK_DIR))

from make_robot import RobotPointCloudGenerator  # noqa: E402
from robot_aware_encoder import (  # noqa: E402
    ROBOT_PC_PRECOMPUTE_VARIANTS,
    robot_pc_zarr_keys,
)

TASKS_32 = [
    "beat_block_hammer",
    "handover_block",
    "hanging_mug",
    "move_can_pot",
    "open_laptop",
    "open_microwave",
    "pick_diverse_bottles",
    "pick_dual_bottles",
    "place_a2b_left",
    "place_a2b_right",
    "place_bread_basket",
    "place_bread_skillet",
    "place_burger_fries",
    "place_cans_plasticbox",
    "place_container_plate",
    "place_dual_shoes",
    "place_empty_cup",
    "place_fan",
    "place_object_basket",
    "place_object_stand",
    "place_phone_stand",
    "place_shoe",
    "press_stapler",
    "put_bottles_dustbin",
    "rotate_qrcode",
    "scan_object",
    "shake_bottle",
    "stack_blocks_three",
    "stack_blocks_two",
    "stack_bowls_three",
    "stack_bowls_two",
    "turn_switch",
]

def _zarr_path(task: str, config: str, n: int) -> Path:
    return Path(__file__).resolve().parents[1] / "data" / f"{task}-{config}-{n}.zarr"

def _write_array(group, name: str, data: np.ndarray, compressor, overwrite: bool):
    if name in group:
        if not overwrite:
            print(f"  skip existing {name} {group[name].shape}")
            return False
        del group[name]
    chunks = (min(100, data.shape[0]),) + data.shape[1:]
    group.create_dataset(
        name,
        data=data,
        chunks=chunks,
        dtype="float32",
        overwrite=True,
        compressor=compressor,
    )
    print(f"  wrote {name} {data.shape}")
    return True

@torch.no_grad()
def precompute_one_variant(
    state: np.ndarray,
    region: str,
    m: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    gen = RobotPointCloudGenerator(
        points_per_link=m,
        sample_region=region,
        device=device,
    )
    T = state.shape[0]
    geo_n = gen.n_links_per_arm * m
    topo_k = 2 * gen.x

    left_geo = np.empty((T, geo_n, 3), dtype=np.float32)
    right_geo = np.empty((T, geo_n, 3), dtype=np.float32)
    left_topo = np.empty((T, topo_k, m, 5), dtype=np.float32)
    right_topo = np.empty((T, topo_k, m, 5), dtype=np.float32)

    t0 = time.time()
    for start in range(0, T, batch_size):
        end = min(start + batch_size, T)
        q = torch.as_tensor(state[start:end], dtype=torch.float32, device=device)
        out = gen.make_robot(q)
        left_geo[start:end] = out["left_geometry_pc"].detach().cpu().numpy()
        right_geo[start:end] = out["right_geometry_pc"].detach().cpu().numpy()
        left_topo[start:end] = out["left_topology_pc"].detach().cpu().numpy()
        right_topo[start:end] = out["right_topology_pc"].detach().cpu().numpy()
        if (start // batch_size) % 20 == 0 or end == T:
            print(f"    {region}_m{m}: {end}/{T} ({time.time()-t0:.1f}s)", flush=True)

    return {
        "left_geometry_pc": left_geo,
        "right_geometry_pc": right_geo,
        "left_topology_pc": left_topo,
        "right_topology_pc": right_topo,
    }

def precompute_zarr(
    zarr_path: Path,
    variants,
    batch_size: int,
    device: str,
    overwrite: bool,
):
    if not zarr_path.exists():
        raise FileNotFoundError(f"zarr not found: {zarr_path}")

    root = zarr.open(str(zarr_path), mode="a")
    data = root["data"]
    state = np.asarray(data["state"][:], dtype=np.float32)
    print(f"[precompute] {zarr_path.name}  T={state.shape[0]}")

    compressor = Blosc(cname="zstd", clevel=3, shuffle=1)
    torch_device = torch.device(device)

    for region, m in variants:
        keys = robot_pc_zarr_keys(region, m)
        if not overwrite and all(k in data for k in keys.values()):
            print(f"  skip {region}_m{m} (all keys present)")
            continue
        arrays = precompute_one_variant(state, region, m, batch_size, torch_device)
        for logical, zkey in keys.items():
            _write_array(data, zkey, arrays[logical], compressor, overwrite=True)

    if "robot_pc_variants" in root.get("meta", {}):
        del root["meta"]["robot_pc_variants"]
    root["meta"].create_dataset(
        "robot_pc_variants",
        data=np.array([f"{r}_m{m}" for r, m in variants], dtype="U32"),
        overwrite=True,
    )
    print(f"[precompute] done {zarr_path.name}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("task_name", nargs="?", default=None)
    parser.add_argument("task_config", type=str, default="demo_clean")
    parser.add_argument("expert_data_num", type=int, default=100)
    parser.add_argument("--all", action="store_true", help="precompute all 32 tasks")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--variants",
        type=str,
        default="all",
        help="all | surface16 | surface8,surface16,inner16 ...",
    )
    args = parser.parse_args()

    if args.variants == "all":
        variants = list(ROBOT_PC_PRECOMPUTE_VARIANTS)
    elif args.variants == "surface16":
        variants = [("surface", 16)]
    else:
        variants = []
        for tok in args.variants.split(","):
            tok = tok.strip()
            if tok.startswith("surface"):
                variants.append(("surface", int(tok.replace("surface", "") or "16")))
            elif tok.startswith("inner"):
                variants.append(("inner", int(tok.replace("inner", "") or "16")))
            else:
                raise ValueError(f"bad variant token: {tok}")

    if args.all:
        tasks = TASKS_32
    else:
        if not args.task_name:
            parser.error("task_name required unless --all")
        tasks = [args.task_name]

    for task in tasks:
        path = _zarr_path(task, args.task_config, args.expert_data_num)
        try:
            precompute_zarr(
                path,
                variants=variants,
                batch_size=args.batch_size,
                device=args.device,
                overwrite=args.overwrite,
            )
        except FileNotFoundError as e:
            print(f"[precompute] WARN {e}")

if __name__ == "__main__":
    main()
