#!/usr/bin/env python3
"""Single-GPU worker for save_dino with multi-step DINO batching. Run from PPI root."""
import argparse
import glob
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import importlib.util

_spec = importlib.util.spec_from_file_location(
    "save_dino", os.path.join(ROOT, "scripts/data_generation/save_dino.py")
)
sd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sd)

CAMERA_NAME = [
    "over_shoulder_left",
    "over_shoulder_right",
    "overhead",
    "wrist_right",
    "wrist_left",
    "front",
]
NUM_CAM = 6
PARAMS = {"patch_h": 64, "patch_w": 64}

def episode_done(target_path, episode, ptc_type, expected_steps=None):
    d = os.path.join(target_path, f"episode{episode}", ptc_type)
    if not os.path.isdir(d):
        return False
    n = len(glob.glob(os.path.join(d, "step*.npy")))
    if expected_steps is None:
        return n > 0
    return n >= expected_steps

def load_step_obs(episode_folder, low_dim_obs, step_i, camera_name):
    depth_image_lst = []
    cameras_extrinsics = []
    cameras_intrinsics = []
    all_cam_images = []
    for camera in camera_name:
        depth_folder = os.path.join(episode_folder, f"{camera}_depth")
        cameras_extrinsics.append(
            sd.transform_camera_extrinsics(
                low_dim_obs[step_i].misc[f"{camera}_camera_extrinsics"][:3, :]
            )
        )
        cameras_intrinsics.append(low_dim_obs[step_i].misc[f"{camera}_camera_intrinsics"])
        near = low_dim_obs[step_i].misc[f"{camera}_camera_near"]
        far = low_dim_obs[step_i].misc[f"{camera}_camera_far"]
        depth_image_rgb = np.array(
            Image.open(os.path.join(depth_folder, f"depth_{step_i:04d}.png"))
        )
        depth_image = sd.rgb_image_to_float_array(depth_image_rgb, 2**24 - 1)
        depth_image_lst.append(near + depth_image * (far - near))
        image = np.array(
            Image.open(
                os.path.join(episode_folder, f"{camera}_rgb", f"rgb_{step_i:04d}.png")
            )
        )
        all_cam_images.append(image)
    return {
        "color": np.stack(all_cam_images, axis=0),
        "depth": np.stack(depth_image_lst, axis=0),
        "pose": np.stack(cameras_extrinsics, axis=0),
        "K": np.stack(cameras_intrinsics, axis=0),
    }

def project_with_precomputed_dino(fusion, ptc, obs, dino_feats):
    """Same as extract_semantic_feature_from_ptc but reuse batched DINO features."""
    fusion.num_cam = obs["color"].shape[0]
    fusion.curr_obs_torch["dino_feats"] = dino_feats
    fusion.curr_obs_torch["color"] = obs["color"]
    fusion.curr_obs_torch["color_tensor"] = (
        torch.from_numpy(obs["color"]).to(fusion.device, dtype=fusion.dtype) / 255.0
    )
    fusion.curr_obs_torch["depth"] = torch.from_numpy(obs["depth"]).to(
        fusion.device, dtype=fusion.dtype
    )
    fusion.curr_obs_torch["pose"] = torch.from_numpy(obs["pose"]).to(
        fusion.device, dtype=fusion.dtype
    )
    fusion.curr_obs_torch["K"] = torch.from_numpy(obs["K"]).to(
        fusion.device, dtype=fusion.dtype
    )
    _, fusion.H, fusion.W = obs["depth"].shape
    return fusion.eval(ptc, return_names=["dino_feats"], return_inter=False)["dino_feats"]

def process_episodes_resume(
    begin, end, task, fusion, pcd_path, data_path, target_path, ptc_type, device, batch_steps
):
    for episode in range(begin, end + 1):
        episode_folder = os.path.join(data_path, f"episode{episode}")
        low_dim_obs_path = os.path.join(episode_folder, "low_dim_obs.pkl")
        if not os.path.isfile(low_dim_obs_path):
            print(f"[skip] missing {low_dim_obs_path}", flush=True)
            continue
        low_dim_obs = sd.read_pkl(low_dim_obs_path)
        n_steps = len(low_dim_obs)
        out_dir = os.path.join(target_path, f"episode{episode}", ptc_type)
        if episode_done(target_path, episode, ptc_type, expected_steps=n_steps):
            print(f"[skip] episode{episode} already has {n_steps} steps", flush=True)
            continue
        os.makedirs(out_dir, exist_ok=True)
        pcd_folder = os.path.join(pcd_path, f"episode{episode}/{ptc_type}")
        t0 = time.time()
        i = 0
        while i < n_steps:
            # gather a contiguous chunk of missing steps (up to batch_steps)
            chunk = []
            while i < n_steps and len(chunk) < batch_steps:
                out_f = os.path.join(out_dir, f"step{i:03d}.npy")
                if os.path.isfile(out_f):
                    i += 1
                    continue
                chunk.append(i)
                i += 1
            if not chunk:
                continue

            colors = []
            obss = []
            ptcs = []
            for si in chunk:
                obs = load_step_obs(episode_folder, low_dim_obs, si, CAMERA_NAME)
                obss.append(obs)
                colors.append(obs["color"])
                ptcs.append(
                    torch.tensor(
                        np.load(os.path.join(pcd_folder, f"step{si:03d}.npy"))[..., :3],
                        dtype=torch.float32,
                        device=device,
                    )
                )

            # Batched DINO: (B*6, H, W, 3) — LayerNorm, equivalent to per-step
            color_batch = np.concatenate(colors, axis=0)
            try:
                feats_all = fusion.extract_features(color_batch, PARAMS)
            except RuntimeError as e:
                if "out of memory" not in str(e).lower():
                    raise
                torch.cuda.empty_cache()
                print(
                    f"[OOM] batch_steps={len(chunk)} failed, fallback per-step",
                    flush=True,
                )
                feats_all = None

            for bi, si in enumerate(chunk):
                out_f = os.path.join(out_dir, f"step{si:03d}.npy")
                if feats_all is None:
                    feat = fusion.extract_semantic_feature_from_ptc(ptcs[bi], obss[bi])
                else:
                    dino_feats = feats_all[bi * NUM_CAM : (bi + 1) * NUM_CAM]
                    feat = project_with_precomputed_dino(
                        fusion, ptcs[bi], obss[bi], dino_feats
                    )
                np.save(out_f, feat.detach().cpu().numpy())

            last = chunk[-1]
            if (last + 1) % 20 < len(chunk) or last == n_steps - 1:
                mem = torch.cuda.memory_allocated(device) / (1024**3)
                peak = torch.cuda.max_memory_allocated(device) / (1024**3)
                print(
                    f"[{task}] ep{episode} step {last+1}/{n_steps} "
                    f"batch={len(chunk)} imgs={len(chunk)*NUM_CAM} "
                    f"elapsed={time.time()-t0:.1f}s "
                    f"vram={mem:.2f}G peak={peak:.2f}G",
                    flush=True,
                )
        print(
            f"[{task}] Episode {episode} finished in {time.time()-t0:.1f}s "
            f"(batch_steps={batch_steps})",
            flush=True,
        )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=99)
    ap.add_argument(
        "--batch-steps",
        type=int,
        default=int(os.environ.get("BATCH_STEPS", "16")),
        help="Timesteps per DINO forward (images = batch_steps * 6)",
    )
    args = ap.parse_args()

    task = args.task
    device = args.device
    pcd_path = f"data/training_processed/point_cloud/{task}/all_variations/episodes"
    data_path = f"data/training_raw/{task}/all_variations/episodes"
    target_path = f"data/training_processed/dino_feature/{task}/all_variations/episodes"
    ptc_type = "rgb_pcd_rps6144"
    os.makedirs(target_path, exist_ok=True)

    print(
        f"START task={task} device={device} eps={args.start}-{args.end} "
        f"batch_steps={args.batch_steps} dino_imgs={args.batch_steps * NUM_CAM}",
        flush=True,
    )
    fusion = sd.Fusion(num_cam=6, feat_backbone="dinov2", device=device)
    torch.cuda.reset_peak_memory_stats(device)
    process_episodes_resume(
        args.start,
        args.end,
        task,
        fusion,
        pcd_path,
        data_path,
        target_path,
        ptc_type,
        device,
        args.batch_steps,
    )
    print(f"DONE task={task} device={device} eps={args.start}-{args.end}", flush=True)

if __name__ == "__main__":
    main()
