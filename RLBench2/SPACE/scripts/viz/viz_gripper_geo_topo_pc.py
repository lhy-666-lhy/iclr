#!/usr/bin/env python3
"""Visualize DualPanda gripper geometry + palm-frame topology PCs (RLBench2).

Geometry (world): table/scene + dual-arm colored links, black background.
Topology (palm frame): palm=blue, fingers=green, other links=gray.

Theme: Collisions with Tiny Geometric Structures

Example:
  COPPELIASIM_ROOT=$COPPELIASIM_ROOT \\
  python scripts/viz/viz_gripper_geo_topo_pc.py \\
      --episode 80 --step 61 --points_per_link 400
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

PPI_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PPI_ROOT))

from space.utils.dual_panda_robot_pc import (  # noqa: E402
    ARM_SHAPES,
    DUAL_PANDA_LINK_NAMES,
    TOPOLOGY_PAIRS,
    apply_demo_to_dual_panda,
    build_topology_pc,
    sample_arm_from_shapes,
)

OUT_DIR = PPI_ROOT / "exp_logs" / "fail_videos_handover_easy" / "gripper_geo_topo_viz"
DEFAULT_SCENE = Path(
    os.environ.get("COPPELIASIM_ROOT", "")
    "rlbench/task_design_bimanual.ttt"
)

# World geometry: distinct per-link colors (RGB 0-1)
LINK_COLORS = {
    "base": (0.55, 0.55, 0.55),
    "link0": (0.65, 0.65, 0.65),
    "link1": (0.90, 0.10, 0.30),
    "link2": (0.24, 0.71, 0.29),
    "link3": (1.00, 0.88, 0.10),
    "link4": (0.00, 0.51, 0.78),
    "link5": (0.96, 0.51, 0.19),
    "link6": (0.57, 0.12, 0.71),
    "link7": (0.27, 0.94, 0.94),
    "gripper": (0.20, 0.45, 1.00),  # palm-ish blue in world too
    "finger1": (0.15, 0.85, 0.25),
    "finger2": (0.35, 0.95, 0.40),
}

PALM_BLUE = (0.20, 0.45, 1.00)
FINGER_GREEN = (0.15, 0.85, 0.25)
OTHER_GRAY = (0.55, 0.55, 0.55)

GRIPPER_ID = DUAL_PANDA_LINK_NAMES.index("gripper")  # 9
FINGER_IDS = {
    DUAL_PANDA_LINK_NAMES.index("finger1"),
    DUAL_PANDA_LINK_NAMES.index("finger2"),
}

def _invert_se3(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out

def _transform_points(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    if pts.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    ones = np.ones((pts.shape[0], 1), dtype=np.float64)
    return (T @ np.concatenate([pts, ones], axis=1).T).T[:, :3]

def load_demo_obs(task: str, episode: int, step: int):
    p = (
        PPI_ROOT
        / "data/training_raw"
        / task
        / "all_variations/episodes"
        / f"episode{episode}"
        / "low_dim_obs.pkl"
    )
    demo = pickle.load(open(p, "rb"))
    return demo[step]

def load_scene_pc(task: str, episode: int, step: int) -> Tuple[np.ndarray, np.ndarray]:
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
    # slightly darken scene so robot pops on black bg (keep visible)
    rgb = np.clip(rgb * 0.75 + 0.12, 0, 1)
    return xyz, rgb

def sample_dense_from_pyrep(
    demo_obs,
    scene_ttt: Path,
    points_per_link: int,
) -> Dict[str, object]:
    from pyrep import PyRep
    from pyrep.objects.shape import Shape
    from pyrep.robots.arms.dual_panda import PandaLeft, PandaRight

    pr = PyRep()
    pr.launch(str(scene_ttt), headless=True)
    pr.start()
    try:
        left, right = PandaLeft(), PandaRight()
        apply_demo_to_dual_panda(demo_obs, left, right)
        pr.step()

        rng = np.random.default_rng(0)
        arm_local: Dict[str, List[np.ndarray]] = {}
        arm_T: Dict[str, List[np.ndarray]] = {}
        arm_world: Dict[str, Dict[str, np.ndarray]] = {}
        arm_geo: Dict[str, np.ndarray] = {}
        arm_topo: Dict[str, np.ndarray] = {}

        for arm, spec in ARM_SHAPES.items():
            shapes = [Shape(name) for _, name in spec]
            geo, _, local_pts, world_T = sample_arm_from_shapes(shapes, points_per_link, rng)
            topo = build_topology_pc(local_pts, world_T, points_per_link)
            arm_local[arm] = local_pts
            arm_T[arm] = world_T
            arm_geo[arm] = geo
            arm_topo[arm] = topo
            arm_world[arm] = {
                name: _transform_points(world_T[i], local_pts[i])
                for i, name in enumerate(DUAL_PANDA_LINK_NAMES)
            }
        return {
            "arm_world": arm_world,
            "arm_local": arm_local,
            "arm_T": arm_T,
            "arm_geo": arm_geo,
            "arm_topo": arm_topo,
        }
    finally:
        pr.stop()
        pr.shutdown()

def points_in_palm_frame(
    local_pts: List[np.ndarray],
    world_T: List[np.ndarray],
) -> Dict[str, np.ndarray]:
    """All link points expressed in gripper (palm) local frame."""
    T_palm_inv = _invert_se3(world_T[GRIPPER_ID])
    out = {}
    for i, name in enumerate(DUAL_PANDA_LINK_NAMES):
        pw = _transform_points(world_T[i], local_pts[i])
        out[name] = _transform_points(T_palm_inv, pw)
    return out

def topology_in_palm_frame(topo: np.ndarray) -> Dict[str, np.ndarray]:
    """Collect topology edge points whose neighbor is the palm (gripper).

    These are already in palm frame by construction of build_topology_pc.
    """
    buckets = {"palm_self": [], "fingers": [], "others": []}
    for k, (self_id, nei_id) in enumerate(TOPOLOGY_PAIRS):
        if nei_id != GRIPPER_ID:
            continue
        pts = topo[k, :, :3]
        if self_id == GRIPPER_ID:
            buckets["palm_self"].append(pts)
        elif self_id in FINGER_IDS:
            buckets["fingers"].append(pts)
        else:
            buckets["others"].append(pts)
    # Also include gripper→* inverted? User wants palm points: use local gripper samples
    # via geometry path instead. Here return concatenated.
    return {
        k: (np.concatenate(v, 0) if v else np.zeros((0, 3)))
        for k, v in buckets.items()
    }

def _set_black_3d(ax):
    ax.set_facecolor("black")
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_edgecolor((0.2, 0.2, 0.2))
        axis.line.set_color((0.5, 0.5, 0.5))
        axis.set_tick_params(colors="0.7", labelsize=7)
    ax.tick_params(colors="0.7")
    ax.xaxis.label.set_color("0.8")
    ax.yaxis.label.set_color("0.8")
    ax.zaxis.label.set_color("0.8")
    ax.title.set_color("white")

def _equal_aspect(ax, pts: np.ndarray, pad: float = 0.02):
    if pts.size == 0:
        return
    c = pts.mean(0)
    r = max(float(np.abs(pts - c).max()) * 1.15, pad)
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(c[2] - r, c[2] + r)

def render_geometry_png(
    scene_xyz,
    scene_rgb,
    arm_world: Dict[str, Dict[str, np.ndarray]],
    out_path: Path,
    title: str,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(11, 8), facecolor="black")
    ax = fig.add_subplot(111, projection="3d")
    _set_black_3d(ax)

    n = len(scene_xyz)
    idx = np.arange(n) if n <= 5000 else np.random.default_rng(0).choice(n, 5000, replace=False)
    ax.scatter(
        scene_xyz[idx, 0],
        scene_xyz[idx, 1],
        scene_xyz[idx, 2],
        c=scene_rgb[idx],
        s=1.5,
        alpha=0.35,
        linewidths=0,
    )

    all_pts = [scene_xyz[idx]]
    for arm, links in arm_world.items():
        for name, pts in links.items():
            if pts.size == 0:
                continue
            c = LINK_COLORS[name]
            if arm == "right":
                c = tuple(float(np.clip(v * t, 0, 1)) for v, t in zip(c, (0.85, 0.95, 1.15)))
            s = 10 if name in ("gripper", "finger1", "finger2") else 4
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=[c], s=s, alpha=0.95, linewidths=0)
            all_pts.append(pts)

    stack = np.concatenate(all_pts, 0)
    _equal_aspect(ax, stack)
    ax.view_init(elev=22, azim=-55)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(title, fontsize=12, pad=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, facecolor="black", bbox_inches="tight")
    plt.close(fig)

def _crop_near_origin(pts: np.ndarray, radius: float) -> np.ndarray:
    if pts.size == 0:
        return pts
    return pts[np.linalg.norm(pts, axis=1) <= radius]

def render_palm_topo_png(
    palm_frames: Dict[str, Dict[str, np.ndarray]],
    out_path: Path,
    title: str,
    focus_radius: float = 0.12,
):
    """Palm-local view focused on tiny gripper structures (palm/fingers + nearby links)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    # Only keep gripper-neighborhood links as "other" (tiny structures context)
    other_keep = ("link6", "link7")

    fig = plt.figure(figsize=(14, 6.5), facecolor="black")
    for pi, (arm, parts) in enumerate(palm_frames.items()):
        ax = fig.add_subplot(1, 2, pi + 1, projection="3d")
        _set_black_3d(ax)
        all_pts = []

        # others gray first (crop to palm neighborhood)
        for name in other_keep:
            pts = _crop_near_origin(parts.get(name, np.zeros((0, 3))), focus_radius)
            if pts.size == 0:
                continue
            ax.scatter(
                pts[:, 0], pts[:, 1], pts[:, 2], c=[OTHER_GRAY], s=8, alpha=0.55, linewidths=0,
                label="other" if name == other_keep[0] else None,
            )
            all_pts.append(pts)

        # palm blue
        palm = _crop_near_origin(parts["gripper"], focus_radius)
        ax.scatter(
            palm[:, 0], palm[:, 1], palm[:, 2], c=[PALM_BLUE], s=18, alpha=0.95, linewidths=0, label="palm"
        )
        all_pts.append(palm)

        # fingers green
        for fi, fname in enumerate(("finger1", "finger2")):
            pts = _crop_near_origin(parts[fname], focus_radius)
            ax.scatter(
                pts[:, 0],
                pts[:, 1],
                pts[:, 2],
                c=[FINGER_GREEN],
                s=18,
                alpha=0.95,
                linewidths=0,
                label="finger" if fi == 0 else None,
            )
            all_pts.append(pts)

        # axes at palm origin
        ax.scatter([0], [0], [0], c="white", s=50, marker="x")
        axis_len = 0.025
        for vec, col in (
            ([axis_len, 0, 0], "r"),
            ([0, axis_len, 0], "g"),
            ([0, 0, axis_len], "b"),
        ):
            ax.plot([0, vec[0]], [0, vec[1]], [0, vec[2]], color=col, lw=1.8)

        stack = np.concatenate(all_pts, 0) if all_pts else np.zeros((1, 3))
        _equal_aspect(ax, stack, pad=0.005)
        ax.view_init(elev=22, azim=35)
        ax.set_title(
            f"{arm} palm frame (zoom ≤{focus_radius:.2f}m)\npalm=blue  fingers=green  other=gray",
            color="white",
        )
        ax.legend(loc="upper left", fontsize=8, facecolor="0.15", labelcolor="white", framealpha=0.8)

    fig.suptitle(title, color="white", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, facecolor="black", bbox_inches="tight")
    plt.close(fig)

def write_html_geometry(
    scene_xyz,
    scene_rgb,
    arm_world,
    out_path: Path,
    title: str,
):
    n = len(scene_xyz)
    idx = np.arange(n) if n <= 6000 else np.random.default_rng(0).choice(n, 6000, replace=False)
    clouds = []
    # scene
    clouds.append(
        {
            "name": "table/scene",
            "xyz": scene_xyz[idx].tolist(),
            "rgb": scene_rgb[idx].tolist(),
            "size": 1.6,
        }
    )
    for arm, links in arm_world.items():
        for name, pts in links.items():
            if pts.size == 0:
                continue
            c = LINK_COLORS[name]
            if arm == "right":
                c = [float(np.clip(v * t, 0, 1)) for v, t in zip(c, (0.85, 0.95, 1.15))]
            else:
                c = list(c)
            clouds.append(
                {
                    "name": f"{arm}/{name}",
                    "xyz": pts.tolist(),
                    "rgb": [c] * len(pts),
                    "size": 3.2 if name in ("gripper", "finger1", "finger2") else 2.0,
                }
            )
    _write_three_html(clouds, out_path, title, bg="#000000")

def write_html_palm(
    palm_frames: Dict[str, Dict[str, np.ndarray]],
    out_path: Path,
    title: str,
    focus_radius: float = 0.12,
):
    clouds = []
    keep = ("link6", "link7", "gripper", "finger1", "finger2")
    # offset right arm in x so both visible side by side
    for arm, parts in palm_frames.items():
        ox = 0.0 if arm == "left" else 0.16
        for name in keep:
            pts = _crop_near_origin(parts.get(name, np.zeros((0, 3))), focus_radius)
            if pts.size == 0:
                continue
            if name == "gripper":
                c = list(PALM_BLUE)
                size = 4.0
            elif name in ("finger1", "finger2"):
                c = list(FINGER_GREEN)
                size = 4.0
            else:
                c = list(OTHER_GRAY)
                size = 2.2
            xyz = pts.copy()
            xyz[:, 0] += ox
            clouds.append(
                {
                    "name": f"{arm}/{name}",
                    "xyz": xyz.tolist(),
                    "rgb": [c] * len(pts),
                    "size": size,
                }
            )
    _write_three_html(clouds, out_path, title, bg="#000000")

def _write_three_html(clouds, out_path: Path, title: str, bg: str = "#000000"):
    """Interactive Three.js HTML (ESM + OrbitControls). Auto-frames point cloud."""
    # Precompute center/extent so camera is never pointing into empty space.
    all_xyz = []
    for cloud in clouds:
        if cloud["xyz"]:
            all_xyz.append(np.asarray(cloud["xyz"], dtype=np.float64))
    if all_xyz:
        stack = np.concatenate(all_xyz, 0)
        center = stack.mean(0)
        extent = float(np.max(np.linalg.norm(stack - center, axis=1)) + 1e-3)
    else:
        center = np.zeros(3)
        extent = 0.3

    payload = {
        "title": title,
        "bg": bg,
        "center": center.tolist(),
        "extent": extent,
        "clouds": clouds,
    }
    data_json = json.dumps(payload)
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>{title}</title>
<style>
  html,body{{margin:0;height:100%;overflow:hidden;background:{bg};color:#e8e8e8;font-family:system-ui,sans-serif}}
  #hud{{position:fixed;left:12px;top:10px;z-index:2;background:rgba(0,0,0,.6);
    padding:10px 14px;border-radius:8px;font-size:13px;max-width:520px;line-height:1.35}}
  #err{{position:fixed;left:12px;bottom:12px;z-index:3;color:#ff8080;font-size:12px;
    background:rgba(0,0,0,.7);padding:8px 10px;border-radius:6px;display:none;max-width:80%}}
</style>
<script type="importmap">
{{
  "imports": {{
    "three": "https://unpkg.com/three@0.160.0/build/three.module.js",
    "three/addons/": "https://unpkg.com/three@0.160.0/examples/jsm/"
  }}
}}
</script>
</head>
<body>
<div id="hud"><b>{title}</b><br/>drag = orbit · wheel = zoom · right-drag = pan<br/>
bg: black · points: square sprites</div>
<div id="err"></div>
<script type="module">
import * as THREE from 'three';
import {{ OrbitControls }} from 'three/addons/controls/OrbitControls.js';

const DATA = {data_json};
const errEl = document.getElementById('err');
window.addEventListener('error', (e) => {{
  errEl.style.display = 'block';
  errEl.textContent = 'JS error: ' + (e.message || e);
}});

const CENTER = new THREE.Vector3(...DATA.center);
const EXTENT = Math.max(DATA.extent, 0.05);

const scene = new THREE.Scene();
scene.background = new THREE.Color(DATA.bg || '#000000');

const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.005, 100);
camera.position.set(CENTER.x + EXTENT * 0.9, CENTER.y - EXTENT * 1.1, CENTER.z + EXTENT * 0.85);

const renderer = new THREE.WebGLRenderer({{ antialias: true }});
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.target.copy(CENTER);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.update();

function makeSquareSprite() {{
  const c = document.createElement('canvas');
  c.width = c.height = 32;
  const ctx = c.getContext('2d');
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(4, 4, 24, 24);
  const tex = new THREE.CanvasTexture(c);
  tex.magFilter = THREE.NearestFilter;
  tex.minFilter = THREE.NearestFilter;
  return tex;
}}
const sprite = makeSquareSprite();

function addCloud(cloud) {{
  const n = cloud.xyz.length;
  if (!n) return;
  const pos = new Float32Array(n * 3);
  const col = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {{
    pos[i*3] = cloud.xyz[i][0];
    pos[i*3+1] = cloud.xyz[i][1];
    pos[i*3+2] = cloud.xyz[i][2];
    col[i*3] = cloud.rgb[i][0];
    col[i*3+1] = cloud.rgb[i][1];
    col[i*3+2] = cloud.rgb[i][2];
  }}
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  geo.setAttribute('color', new THREE.BufferAttribute(col, 3));
  // cloud.size is ~1.6..4; map to meters so points stay visible at arm scale
  const worldSize = Math.max(0.004, (cloud.size || 2.0) * 0.0035);
  const mat = new THREE.PointsMaterial({{
    size: worldSize,
    map: sprite,
    alphaTest: 0.4,
    transparent: true,
    vertexColors: true,
    sizeAttenuation: true,
    depthWrite: true,
  }});
  scene.add(new THREE.Points(geo, mat));
}}
DATA.clouds.forEach(addCloud);

const axes = new THREE.AxesHelper(EXTENT * 0.25);
axes.position.copy(CENTER);
scene.add(axes);

window.addEventListener('resize', () => {{
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
}});

function animate() {{
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}}
animate();
</script>
</body></html>
"""
    out_path.write_text(html, encoding="utf-8")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="bimanual_handover_item_easy")
    ap.add_argument("--episode", type=int, default=80)
    ap.add_argument("--step", type=int, default=61)
    ap.add_argument("--points_per_link", type=int, default=400)
    ap.add_argument("--scene_ttt", type=str, default=str(DEFAULT_SCENE))
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"ep{args.episode}_step{args.step:03d}_m{args.points_per_link}"

    demo_obs = load_demo_obs(args.task, args.episode, args.step)
    scene_xyz, scene_rgb = load_scene_pc(args.task, args.episode, args.step)
    print(f"[info] sampling dense DualPanda PC via PyRep (M={args.points_per_link}) ...")
    packed = sample_dense_from_pyrep(demo_obs, Path(args.scene_ttt), args.points_per_link)
    arm_world = packed["arm_world"]

    palm_frames = {
        arm: points_in_palm_frame(packed["arm_local"][arm], packed["arm_T"][arm])
        for arm in ("left", "right")
    }

    geo_title = (
        f"Collisions with Tiny Geometric Structures — Geometry PC (world)\n"
        f"{args.task} | {tag} | dual-arm colored · table scene · black bg"
    )
    topo_title = (
        f"Collisions with Tiny Geometric Structures — Topology in Palm Frame\n"
        f"{args.task} | {tag} | palm=blue · fingers=green · other=gray"
    )

    geo_png = OUT_DIR / f"geometry_world_{tag}.png"
    topo_png = OUT_DIR / f"topology_palm_frame_{tag}.png"
    geo_html = OUT_DIR / f"geometry_world_{tag}.html"
    topo_html = OUT_DIR / f"topology_palm_frame_{tag}.html"

    render_geometry_png(scene_xyz, scene_rgb, arm_world, geo_png, geo_title)
    render_palm_topo_png(palm_frames, topo_png, topo_title)
    write_html_geometry(scene_xyz, scene_rgb, arm_world, geo_html, geo_title.replace("\n", " | "))
    write_html_palm(palm_frames, topo_html, topo_title.replace("\n", " | "))

    # also dump npy for reuse
    np.savez_compressed(
        OUT_DIR / f"clouds_{tag}.npz",
        left_geo=packed["arm_geo"]["left"],
        right_geo=packed["arm_geo"]["right"],
        left_topo=packed["arm_topo"]["left"],
        right_topo=packed["arm_topo"]["right"],
        **{f"left_palm_{k}": v for k, v in palm_frames["left"].items()},
        **{f"right_palm_{k}": v for k, v in palm_frames["right"].items()},
    )

    # print topology edges into palm for sanity
    for arm in ("left", "right"):
        buckets = topology_in_palm_frame(packed["arm_topo"][arm])
        print(
            f"[topo→palm {arm}] fingers={len(buckets['fingers'])} "
            f"others={len(buckets['others'])} "
            f"(geometry palm pts used for blue palm cloud)"
        )

    print(f"[ok] {geo_png}")
    print(f"[ok] {topo_png}")
    print(f"[ok] {geo_html}")
    print(f"[ok] {topo_html}")
    print(f"[ok] OUT_DIR={OUT_DIR}")

if __name__ == "__main__":
    main()
