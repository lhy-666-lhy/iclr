#!/usr/bin/env python3
"""GT replay + lateral handover bump; record cinematic (main) camera RGB video.

Requires a display (Xvfb ok). Must use headless=False — VisionSensor OpenGL
segfaults in headless=True on this machine.

Example:
  export DISPLAY=:99
  export COPPELIASIM_ROOT=.../CoppeliaSim
  export LD_LIBRARY_PATH=$COPPELIASIM_ROOT:$LD_LIBRARY_PATH
  export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
  export QT_QPA_PLATFORM=xcb
  python scripts/record_handover_bump_fail.py --episode 0 --offset 0.030
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import imageio
import numpy as np
from pyrep.const import RenderMode
from pyrep.objects.dummy import Dummy
from pyrep.objects.shape import Shape
from pyrep.objects.vision_sensor import VisionSensor
from rlbench import Environment
from rlbench.action_modes.action_mode import BimanualJointPositionActionMode
from rlbench.bimanual_tasks.bimanual_handover_item_easy import BimanualHandoverItemEasy
from rlbench.observation_config import ObservationConfig

PPI_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = PPI_ROOT / "exp_logs/fail_videos_handover_easy/gt_bump"
FPS = 20
CAM_RES = [640, 360]  # 1280x720 is too slow / can hang on Xvfb GL

def load_demo(episode: int):
    path = (
        PPI_ROOT
        / "data/training_raw/bimanual_handover_item_easy/all_variations/episodes"
        / f"episode{episode}/low_dim_obs.pkl"
    )
    with open(path, "rb") as f:
        return pickle.load(f), path

def demo_action(obs) -> np.ndarray:
    r_q = np.asarray(obs.right.joint_positions, dtype=np.float64)
    l_q = np.asarray(obs.left.joint_positions, dtype=np.float64)
    r_g = float(np.asarray(obs.right.gripper_open).reshape(-1)[0])
    l_g = float(np.asarray(obs.left.gripper_open).reshape(-1)[0])
    return np.concatenate([r_q, [r_g], l_q, [l_g]])

def set_arms_grippers(robot, obs, disable_dyn: bool = True):
    robot.right_arm.set_joint_positions(
        list(np.asarray(obs.right.joint_positions, dtype=float)), disable_dynamics=disable_dyn
    )
    robot.left_arm.set_joint_positions(
        list(np.asarray(obs.left.joint_positions, dtype=float)), disable_dynamics=disable_dyn
    )
    try:
        robot.right_gripper.set_joint_positions(
            list(np.asarray(obs.right.gripper_joint_positions, dtype=float)),
            disable_dynamics=disable_dyn,
        )
        robot.left_gripper.set_joint_positions(
            list(np.asarray(obs.left.gripper_joint_positions, dtype=float)),
            disable_dynamics=disable_dyn,
        )
    except Exception:
        pass

def set_item_from_demo(obs):
    item = Shape("item")
    pos = np.asarray(obs.object_6d_pose["position"], dtype=np.float64).reshape(-1)[:3]
    quat = np.asarray(obs.object_6d_pose["quaternion"], dtype=np.float64).reshape(-1)[:4]
    item.set_pose(list(pos) + list(quat))

def detect_grasp_side(demo) -> str:
    L = np.array([o.left.gripper_open for o in demo])
    R = np.array([o.right.gripper_open for o in demo])
    first_L = int(np.argmax(L < 0.5)) if (L < 0.5).any() else 10**9
    first_R = int(np.argmax(R < 0.5)) if (R < 0.5).any() else 10**9
    return "L" if first_L <= first_R else "R"

def receiving_opening_axis_world(obs, grasp_side: str) -> np.ndarray:
    recv = obs.right if grasp_side == "L" else obs.left
    M = np.asarray(recv.gripper_matrix, dtype=np.float64)
    y = M[:3, 1].copy()
    return y / (np.linalg.norm(y) + 1e-12)

def ik_offset_holding_arm(robot, obs, grasp_side: str, offset_world: np.ndarray) -> np.ndarray:
    action = demo_action(obs)
    arm = robot.left_arm if grasp_side == "L" else robot.right_arm
    q_gt = obs.left.joint_positions if grasp_side == "L" else obs.right.joint_positions
    arm.set_joint_positions(list(np.asarray(q_gt, dtype=float)), disable_dynamics=True)
    pose = np.asarray(arm.get_tip().get_pose(), dtype=np.float64)
    pose[:3] = pose[:3] + offset_world
    try:
        q = arm.solve_ik_via_jacobian(pose[:3].tolist(), quaternion=pose[3:].tolist())
        if q is not None and len(q) == 7:
            q = np.asarray(q, dtype=np.float64)
            if grasp_side == "L":
                action[8:15] = q
            else:
                action[0:7] = q
            return action
    except Exception as e:
        print(f"[warn] IK failed: {e}")
    if grasp_side == "L":
        action[13] += 0.10
        action[14] -= 0.06
    else:
        action[5] += 0.10
        action[6] -= 0.06
    return action

def force_hold_grasp(robot, grasp_side: str) -> bool:
    item = Shape("item")
    grip = robot.left_gripper if grasp_side == "L" else robot.right_gripper
    ok = bool(grip.grasp(item))
    if not ok:
        tip = (robot.left_arm if grasp_side == "L" else robot.right_arm).get_tip()
        item.set_parent(tip)
        print("[warn] proximity grasp miss; forced parent to tip")
        return True
    print("[ok] grasp(item) attached")
    return True

def setup_main_cam() -> tuple[VisionSensor, Dummy]:
    """Cinematic / main view (same as eval fail videos)."""
    cam_placeholder = Dummy("cam_cinematic_placeholder")
    cam_base = Dummy("cam_cinematic_base")
    cam = VisionSensor.create(CAM_RES)
    cam.set_pose(cam_placeholder.get_pose())
    cam.set_parent(cam_placeholder)
    cam.set_explicit_handling(True)
    cam.set_render_mode(RenderMode.OPENGL)
    return cam, cam_base

def snap(cam: VisionSensor, frames: list, cam_base: Dummy | None = None):
    if cam_base is not None:
        cam_base.rotate([0, 0, 0.003])
    cam.handle_explicitly()
    rgb = (cam.capture_rgb() * 255.0).astype(np.uint8)
    frames.append(rgb)

def item_recv_metrics(robot, scene, grasp_side: str):
    item = np.asarray(Shape("item").get_position(), dtype=np.float64)
    tip_l = np.asarray(robot.left_arm.get_tip().get_position(), dtype=np.float64)
    tip_r = np.asarray(robot.right_arm.get_tip().get_position(), dtype=np.float64)
    live = scene.get_observation()
    M = np.asarray(
        (live.right if grasp_side == "L" else live.left).gripper_matrix, dtype=np.float64
    )
    rel = M[:3, :3].T @ (item - M[:3, 3])
    hold = tip_l if grasp_side == "L" else tip_r
    recv = tip_r if grasp_side == "L" else tip_l
    return {
        "open": float(rel[1]),
        "z": float(rel[2]),
        "d_hold": float(np.linalg.norm(item - hold)),
        "d_recv": float(np.linalg.norm(item - recv)),
        "tip": float(np.linalg.norm(tip_l - tip_r)),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--gt_until", type=int, default=200)
    ap.add_argument("--bump_until", type=int, default=235)
    ap.add_argument("--offset", type=float, default=0.030)
    ap.add_argument("--offset_sign", type=float, default=1.0)
    ap.add_argument("--out_dir", type=str, default=str(DEFAULT_OUT))
    ap.add_argument("--hold_frames", type=int, default=18)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    demo, demo_path = load_demo(args.episode)
    grasp_side = detect_grasp_side(demo)
    print(f"Loaded ep{args.episode} len={len(demo)} hold={grasp_side} offset={args.offset*args.offset_sign:+.3f}m")
    print(f"demo={demo_path}")

    obs_config = ObservationConfig()
    obs_config.set_all(False)
    obs_config.set_all_low_dim(True)
    obs_config.gripper_matrix = True
    obs_config.gripper_joint_positions = True

    # CRITICAL: headless=False for VisionSensor RGB on this machine
    env = Environment(
        action_mode=BimanualJointPositionActionMode(),
        dataset_root="",
        obs_config=obs_config,
        headless=False,
        robot_setup="dual_panda",
    )
    env.launch()
    task = env.get_task(BimanualHandoverItemEasy)
    descriptions, _ = task.reset_to_demo(demo)
    print("reset_to_demo:", descriptions[0] if descriptions else "")

    robot = env._scene.robot
    scene = env._scene
    cam, cam_base = setup_main_cam()
    frames: list[np.ndarray] = []
    logs = []

    gt_until = min(args.gt_until, len(demo) - 1)
    bump_until = min(args.bump_until, len(demo) - 1)

    # --- Phase 1: GT teleport (no/rare capture until near handover) ---
    record_from = max(0, gt_until - 40)
    for t in range(0, gt_until):
        o = demo[t]
        set_arms_grippers(robot, o, disable_dyn=True)
        set_item_from_demo(o)
        scene.step()
        if t >= record_from and t % 2 == 0:
            snap(cam, frames, cam_base=None)  # no orbit — faster/stable
        if t % 40 == 0:
            m = item_recv_metrics(robot, scene, grasp_side)
            print(f"[GT] t={t} d_hold={m['d_hold']:.3f} tip={m['tip']:.3f}", flush=True)

    force_hold_grasp(robot, grasp_side)

    # --- Phase 2: bump with physics ---
    print(f"[bump] t={gt_until}..{bump_until}", flush=True)
    for t in range(gt_until, bump_until + 1):
        o = demo[t]
        alpha = float(np.clip((t - gt_until) / max(1.0, (220 - gt_until)), 0.0, 1.0))
        axis = receiving_opening_axis_world(o, grasp_side)
        off = axis * (args.offset * args.offset_sign * alpha)

        set_arms_grippers(robot, o, disable_dyn=True)
        action = ik_offset_holding_arm(robot, o, grasp_side, off)
        if grasp_side == "L":
            action[15] = 0.0 if t < 218 else 1.0
            action[7] = 1.0 if t < 220 else 0.0
        else:
            action[7] = 0.0 if t < 218 else 1.0
            action[15] = 1.0 if t < 220 else 0.0

        try:
            task.step(action, [])
        except Exception as e:
            print(f"[bump] step fail t={t}: {e}", flush=True)
            robot.right_arm.set_joint_positions(list(action[0:7]), disable_dynamics=False)
            robot.left_arm.set_joint_positions(list(action[8:15]), disable_dynamics=False)
            scene.step()

        snap(cam, frames, cam_base=None)
        m = item_recv_metrics(robot, scene, grasp_side)
        logs.append({"t": t, "alpha": alpha, "off": float(np.linalg.norm(off)), **m})
        if t % 2 == 0 or t in (215, 218, 220, 222, 225):
            print(
                f"[bump] t={t} α={alpha:.2f} |off|={np.linalg.norm(off):.3f} "
                f"open={m['open']:+.3f} z={m['z']:+.3f} d_recv={m['d_recv']:.3f}",
                flush=True,
            )

    for _ in range(args.hold_frames):
        scene.step()
        snap(cam, frames, cam_base=None)

    out_mp4 = out_dir / f"handover_ep{args.episode}_bump_finger_off{args.offset:.3f}_maincam.mp4"
    print(f"Saving {len(frames)} frames -> {out_mp4}")
    imageio.mimsave(str(out_mp4), frames, fps=FPS)

    # contact stills (last ~2s)
    n = len(frames)
    for i in range(max(0, n - 40), n, 2):
        imageio.imwrite(str(out_dir / f"bump_ep{args.episode}_main_f{i:04d}.png"), frames[i])
    imageio.imwrite(str(out_dir / f"bump_ep{args.episode}_main_last.png"), frames[-1])

    with open(out_dir / f"bump_ep{args.episode}_log.json", "w") as f:
        json.dump(logs, f, indent=2)
    if logs:
        best = max(logs, key=lambda r: abs(r["open"]))
        print(
            f"[summary] max |open|={abs(best['open']):.3f} at t={best['t']} "
            f"(z={best['z']:+.3f}, d_recv={best['d_recv']:.3f})"
        )

    env.shutdown()
    print("Done.")
    print(f"OUT={out_dir}")

if __name__ == "__main__":
    main()
