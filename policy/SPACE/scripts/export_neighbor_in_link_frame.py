#!/usr/bin/env python3
"""Export 4 figures: neighbor PC in a base-link local frame.

For each of 4 anchor links:
  - red XYZ arrows only at that link's local origin
  - anchor + one adjacent link: normal per-link colors (same palette as full-arm fig)
  - all other links: gray cubes
  - no green world axes / no ground
  - all points expressed in the anchor link frame

Purpose: show how a neighbor link's surface PC sits in the anchor's local frame.

Usage:
  python policy/SPACE/scripts/export_neighbor_in_link_frame.py \\
      --out_dir policy/SPACE/docs/figures/robot_link_neighbor_frames \\
      --n_points 64
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

os.environ.setdefault("EGL_PLATFORM", "surfaceless")

import open3d as o3d
from open3d.visualization import rendering

SCRIPTS_DIR = Path(__file__).resolve().parent
BLOCK_DIR = Path(__file__).resolve().parents[1] / "block"
for p in (str(SCRIPTS_DIR), str(BLOCK_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from make_robot import (  # noqa: E402
    RobotPointCloudGenerator,
    _invert_se3,
    _transform_local_points,
)
from export_robot_mesh_views import build_demo_qpos  # noqa: E402
from export_full_arm_mesh_axes import (  # noqa: E402
    _axis_group,
    _link_palette,
    _pc_cubes,
    _save_png,
    _sphere_eye,
    COLOR_AXIS,
)

COLOR_GRAY = np.array([0.72, 0.72, 0.72], dtype=np.float64)

# (anchor_link, one_adjacent_link)
DEFAULT_PAIRS: Tuple[Tuple[str, str], ...] = (
    ("fl_link1", "fl_link2"),
    ("fl_link3", "fl_link4"),
    ("fl_link6", "fl_link7"),
    ("fl_link7", "fl_link6"),
)

DEFAULT_VIEW = ("three_quarter", 22.0, 45.0)

def _short(name: str) -> str:
    return name.split("_", 1)[-1] if "_" in name else name

def _clouds_in_anchor_frame(
    gen: RobotPointCloudGenerator,
    qpos: np.ndarray,
    anchor_name: str,
) -> Dict[str, np.ndarray]:
    """All left-arm link surface PCs expressed in ``anchor_name`` frame."""
    if anchor_name not in gen.left_links:
        raise KeyError(anchor_name)
    q = torch.as_tensor(qpos, dtype=torch.float32).unsqueeze(0)
    fk = gen.forward_kinematics(q, arm="left")["left"]
    T_a_inv = _invert_se3(fk[anchor_name])  # (1,4,4)
    out: Dict[str, np.ndarray] = {}
    for name, pts_local in zip(gen.left_links, gen.left_local):
        if name not in fk:
            continue
        p_w = _transform_local_points(fk[name], pts_local)  # (1,M,3)
        ones = torch.ones(1, p_w.shape[1], 1, dtype=p_w.dtype, device=p_w.device)
        p_h = torch.cat([p_w, ones], dim=-1)
        p_a = torch.einsum("bij,bmj->bmi", T_a_inv, p_h)[..., :3][0]
        out[name] = p_a.detach().cpu().numpy().astype(np.float64)
    return out

def _pick_color(
    name: str,
    anchor: str,
    neighbor: str,
    name_to_idx: Dict[str, int],
    palette: np.ndarray,
) -> np.ndarray:
    if name in (anchor, neighbor):
        return palette[name_to_idx[name] % len(palette)]
    return COLOR_GRAY.copy()

def render_one(
    clouds: Dict[str, np.ndarray],
    link_order: Sequence[str],
    anchor: str,
    neighbor: str,
    palette: np.ndarray,
    out_path: Path,
    elev: float,
    azim: float,
    axis_length: float,
    axis_radius: float,
    cube_scale: float,
    width: int,
    height: int,
) -> Path:
    name_to_idx = {n: i for i, n in enumerate(link_order)}
    pts_all = []
    for name in link_order:
        if name in clouds and len(clouds[name]):
            pts_all.append(clouds[name])
    if not pts_all:
        raise RuntimeError(f"no points for anchor={anchor}")
    stack = np.vstack(pts_all)
    # Include origin (axes)
    stack = np.vstack([stack, np.zeros((1, 3))])
    mn, mx = stack.min(0), stack.max(0)
    center = 0.5 * (mn + mx)
    extent = float(np.linalg.norm(mx - mn))
    cam_radius = max(extent * 1.35, 0.12)
    cube_size = max(extent * cube_scale, 0.0025)

    renderer = rendering.OffscreenRenderer(width, height)
    scene = renderer.scene
    scene.set_background([1.0, 1.0, 1.0, 1.0])
    sun_dir = np.array([0.35, -0.25, -0.9], dtype=np.float32)
    scene.set_lighting(rendering.Open3DScene.NO_SHADOWS, sun_dir)

    # Per-link cubes
    for i, name in enumerate(link_order):
        pts = clouds.get(name)
        if pts is None or len(pts) == 0:
            continue
        color = _pick_color(name, anchor, neighbor, name_to_idx, palette)
        cubes = _pc_cubes(pts, color, cube_size)
        mat = rendering.MaterialRecord()
        mat.shader = "defaultLit"
        mat.base_color = [color[0], color[1], color[2], 1.0]
        gname = f"pc_{i}"
        scene.add_geometry(gname, cubes, mat)
        scene.scene.geometry_shadows(gname, False, False)

    # Red axes only for the anchor frame (at origin)
    mat_axis = rendering.MaterialRecord()
    mat_axis.shader = "defaultLit"
    mat_axis.base_color = [COLOR_AXIS[0], COLOR_AXIS[1], COLOR_AXIS[2], 1.0]
    for i, a in enumerate(_axis_group(axis_length, axis_radius, COLOR_AXIS)):
        gname = f"axis_{i}"
        scene.add_geometry(gname, a, mat_axis)
        scene.scene.geometry_shadows(gname, False, False)

    up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    eye = _sphere_eye(center, cam_radius, elev, azim)
    look = center - eye
    look = look / (np.linalg.norm(look) + 1e-9)
    cam_up = up.copy()
    if abs(np.dot(look, up)) > 0.92:
        cam_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    renderer.setup_camera(40.0, center, eye, cam_up)
    img = renderer.render_to_image()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _save_png(out_path, img)
    del renderer
    print(
        f"[saved] {out_path}  "
        f"(anchor={anchor}, neighbor={neighbor}, cube={cube_size:.4f})"
    )
    return out_path

def write_contact_sheet(
    paths: Sequence[Path], titles: Sequence[str], out_path: Path, dpi: int = 300
) -> None:
    try:
        import matplotlib.pyplot as plt
        from PIL import Image
    except Exception as e:
        print(f"[warn] contact sheet skipped: {e}")
        return
    n = len(paths)
    cols = min(2, n)
    rows = int(np.ceil(n / max(cols, 1)))
    fig, axes = plt.subplots(rows, cols, figsize=(5.4 * cols, 4.4 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, p, t in zip(axes, paths, titles):
        ax.imshow(Image.open(p))
        ax.set_title(t, fontsize=10)
        ax.axis("off")
    fig.patch.set_facecolor("white")
    fig.tight_layout(pad=0.4)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_path}")

def parse_pairs(s: Optional[str]) -> List[Tuple[str, str]]:
    if not s:
        return list(DEFAULT_PAIRS)
    pairs = []
    for part in s.split(";"):
        part = part.strip()
        if not part:
            continue
        a, b = [x.strip() for x in part.split(",")]
        pairs.append((a, b))
    return pairs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out_dir",
        type=str,
        default=str(
            Path(__file__).resolve().parents[1]
            / "docs"
            / "figures"
            / "robot_link_neighbor_frames"
        ),
    )
    parser.add_argument(
        "--pairs",
        type=str,
        default="",
        help='Semicolon-separated "anchor,neighbor" pairs; default = 4 paper links',
    )
    parser.add_argument("--n_points", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--elev", type=float, default=DEFAULT_VIEW[1])
    parser.add_argument("--azim", type=float, default=DEFAULT_VIEW[2])
    parser.add_argument("--axis_length", type=float, default=0.055)
    parser.add_argument("--axis_radius", type=float, default=0.0022)
    parser.add_argument("--cube_scale", type=float, default=0.020)
    parser.add_argument("--width", type=int, default=1400)
    parser.add_argument("--height", type=int, default=1100)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    pairs = parse_pairs(args.pairs or None)

    gen = RobotPointCloudGenerator(
        points_per_link=args.n_points,
        sample_region="surface",
        device="cpu",
    )
    qpos = build_demo_qpos(args.seed)
    palette = _link_palette(len(gen.left_links))
    print("left links:", gen.left_links)
    print("pairs:", pairs)

    saved: List[Path] = []
    titles: List[str] = []
    for anchor, neighbor in pairs:
        if anchor not in gen.left_links:
            raise SystemExit(f"anchor {anchor!r} not in {gen.left_links}")
        if neighbor not in gen.left_links:
            raise SystemExit(f"neighbor {neighbor!r} not in {gen.left_links}")
        clouds = _clouds_in_anchor_frame(gen, qpos, anchor)
        path = out_dir / (
            f"neighbor_in_{_short(anchor)}_frame"
            f"_adj_{_short(neighbor)}"
            f"_e{int(args.elev)}_a{int(args.azim)}.png"
        )
        render_one(
            clouds,
            gen.left_links,
            anchor,
            neighbor,
            palette,
            path,
            elev=args.elev,
            azim=args.azim,
            axis_length=args.axis_length,
            axis_radius=args.axis_radius,
            cube_scale=args.cube_scale,
            width=args.width,
            height=args.height,
        )
        saved.append(path)
        titles.append(f"{_short(neighbor)} in {_short(anchor)} frame")

    sheet = out_dir / "00_four_neighbor_frames_contact_sheet.png"
    write_contact_sheet(saved, titles, sheet, dpi=args.dpi)

    readme = out_dir / "README.md"
    lines = [
        "# Neighbor PC in anchor link local frame",
        "",
        "Extra figures (do **not** replace the full-arm world figure).",
        "",
        "- **Red arrows**: only the anchor link local XYZ at the origin",
        "- **Colored cubes**: anchor + one adjacent link (same palette as full-arm fig)",
        "- **Gray cubes**: all other left-arm links",
        "- No green world axes / no ground",
        "",
        f"n_points=**{args.n_points}**, seed={args.seed}, elev={args.elev}, azim={args.azim}",
        "",
        "| File | Anchor frame | Adjacent (colored) |",
        "|------|--------------|--------------------|",
    ]
    for (anchor, neighbor), p in zip(pairs, saved):
        lines.append(f"| `{p.name}` | `{anchor}` | `{neighbor}` |")
    lines.append("")
    lines.append("Also: `00_four_neighbor_frames_contact_sheet.png`.")
    readme.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Done → {out_dir}")

if __name__ == "__main__":
    main()
