#!/usr/bin/env python3
"""Left-arm PC in an anchor link local frame: blue=anchor, orange=neighbor, gray=others.

Same style as docs/figures/robot_pc_link6_frame (dark bg, red axes, multi-view + RECOMMENDED).

Usage:
  # one group (default = link6 / finger)
  python policy/SPACE/scripts/export_robot_pc_link_frame_views.py

  # three extra groups (link1 / link3 / link7)
  python policy/SPACE/scripts/export_robot_pc_link_frame_views.py --batch_three
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

BLOCK_DIR = Path(__file__).resolve().parents[1] / "block"
if str(BLOCK_DIR) not in sys.path:
    sys.path.insert(0, str(BLOCK_DIR))

from make_robot import (  # noqa: E402
    RobotPointCloudGenerator,
    _invert_se3,
    _transform_local_points,
)

COLOR_OTHER = np.array([0.55, 0.55, 0.55], dtype=np.float64)  # gray
COLOR_AXIS = np.array([1.00, 0.08, 0.08], dtype=np.float64)  # red

# Default (legacy link6 style)
COLOR_ANCHOR_DEFAULT = np.array([0.15, 0.45, 1.00], dtype=np.float64)
COLOR_HIGHLIGHT_DEFAULT = np.array([1.00, 0.55, 0.05], dtype=np.float64)

DEFAULT_VIEWS: List[Tuple[str, float, float]] = [
    ("best_A", 28.0, 40.0),
    ("best_B", 32.0, -35.0),
    ("best_C", 18.0, 70.0),
    ("best_D", 22.0, -65.0),
    ("best_E", 45.0, 25.0),
    ("best_F", 12.0, 110.0),
    ("topish", 62.0, 30.0),
    ("side", 15.0, 95.0),
]

# (anchor, highlight, anchor_rgb, highlight_rgb) — distinct pairs per group
BATCH_THREE: Tuple[Tuple[str, str, np.ndarray, np.ndarray], ...] = (
    (
        "fl_link1",
        "fl_link2",
        np.array([0.10, 0.70, 0.85], dtype=np.float64),  # cyan
        np.array([0.95, 0.35, 0.55], dtype=np.float64),  # rose
    ),
    (
        "fl_link3",
        "fl_link4",
        np.array([0.20, 0.72, 0.30], dtype=np.float64),  # green
        np.array([0.62, 0.28, 0.82], dtype=np.float64),  # purple
    ),
    (
        "fl_link7",
        "fl_link6",
        np.array([0.95, 0.45, 0.10], dtype=np.float64),  # orange
        np.array([0.20, 0.45, 0.95], dtype=np.float64),  # blue
    ),
)

def _short(name: str) -> str:
    return name.split("_", 1)[-1] if "_" in name else name

def build_show_qpos() -> np.ndarray:
    q = np.zeros(14, dtype=np.float32)
    q[0] = 0.45
    q[1] = 0.70
    q[2] = 1.05
    q[3] = -0.95
    q[4] = 0.70
    q[5] = 0.40
    q[6] = 0.75
    return q

def _world_clouds(
    gen: RobotPointCloudGenerator, qpos: np.ndarray
) -> Tuple[Dict[str, np.ndarray], Dict[str, torch.Tensor]]:
    q = torch.as_tensor(qpos, dtype=torch.float32).unsqueeze(0)
    fk = gen.forward_kinematics(q, arm="left")["left"]
    clouds: Dict[str, np.ndarray] = {}
    for name, pts_local in zip(gen.left_links, gen.left_local):
        if name not in fk:
            continue
        pw = _transform_local_points(fk[name], pts_local)[0]
        clouds[name] = pw.detach().cpu().numpy().astype(np.float64)
    return clouds, fk

def _to_anchor_frame(
    clouds_w: Dict[str, np.ndarray], T_anchor_w: torch.Tensor
) -> Dict[str, np.ndarray]:
    """p_anchor = R^T (p_w - t)."""
    T_inv = _invert_se3(T_anchor_w)
    if T_inv.ndim == 2:
        T_inv = T_inv.unsqueeze(0)
    out: Dict[str, np.ndarray] = {}
    for name, pts in clouds_w.items():
        t = torch.as_tensor(pts, dtype=torch.float32)
        pa = _transform_local_points(T_inv, t)[0].detach().cpu().numpy().astype(np.float64)
        out[name] = pa
    return out

def _colorize(
    clouds: Dict[str, np.ndarray],
    anchor: str,
    highlight: str,
    color_anchor: np.ndarray,
    color_highlight: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return all pts/cols, focus (anchor+highlight), highlight pts."""
    pts_all: List[np.ndarray] = []
    cols_all: List[np.ndarray] = []
    focus: List[np.ndarray] = []
    highlight_pts = clouds.get(highlight, np.zeros((0, 3)))
    for name, pts in clouds.items():
        if name == anchor:
            c = color_anchor
            focus.append(pts)
        elif name == highlight:
            c = color_highlight
            focus.append(pts)
        else:
            c = COLOR_OTHER
        pts_all.append(pts)
        cols_all.append(np.tile(c, (len(pts), 1)))
    focus_arr = np.concatenate(focus, axis=0) if focus else np.zeros((0, 3))
    return (
        np.concatenate(pts_all, axis=0),
        np.concatenate(cols_all, axis=0),
        focus_arr,
        np.asarray(highlight_pts, dtype=np.float64),
    )

def _axis_arrow(axis: str, length: float, radius: float) -> o3d.geometry.TriangleMesh:
    cone_h = length * 0.18
    cyl_h = length - cone_h
    arrow = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=radius,
        cone_radius=radius * 2.2,
        cylinder_height=cyl_h,
        cone_height=cone_h,
        resolution=20,
        cylinder_split=4,
        cone_split=1,
    )
    arrow.paint_uniform_color(COLOR_AXIS.tolist())
    arrow.compute_vertex_normals()
    if axis == "x":
        R = arrow.get_rotation_matrix_from_xyz((0.0, np.pi / 2, 0.0))
        arrow.rotate(R, center=(0, 0, 0))
    elif axis == "y":
        R = arrow.get_rotation_matrix_from_xyz((-np.pi / 2, 0.0, 0.0))
        arrow.rotate(R, center=(0, 0, 0))
    elif axis == "z":
        pass
    else:
        raise ValueError(axis)
    return arrow

def _axis_group(scale: float) -> List[o3d.geometry.TriangleMesh]:
    r = max(scale * 0.014, 0.0018)
    L = max(scale * 1.25, 0.10)
    return [_axis_arrow("x", L, r), _axis_arrow("y", L, r), _axis_arrow("z", L, r)]

def _sphere_eye(center: np.ndarray, radius: float, elev_deg: float, azim_deg: float) -> np.ndarray:
    elev = np.deg2rad(elev_deg)
    azim = np.deg2rad(azim_deg)
    x = radius * np.cos(elev) * np.cos(azim)
    y = radius * np.cos(elev) * np.sin(azim)
    z = radius * np.sin(elev)
    return center + np.array([x, y, z], dtype=np.float64)

def _view_score(
    eye: np.ndarray, focus_pts: np.ndarray, highlight_pts: np.ndarray, fov_deg: float = 50.0
) -> float:
    if len(highlight_pts) == 0:
        return -1e9
    center = focus_pts.mean(axis=0)
    forward = center - eye
    dist = np.linalg.norm(forward) + 1e-9
    forward = forward / dist
    v = highlight_pts - eye
    vn = np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
    cosang = ((v / vn) * forward).sum(axis=1)
    half = np.cos(np.deg2rad(fov_deg * 0.55))
    in_fov = float((cosang > half).mean())
    dist_term = -abs(
        np.log(dist / (float(np.linalg.norm(np.ptp(focus_pts, axis=0))) + 1e-6))
    )
    return 2.5 * in_fov + 0.15 * dist_term

def render_views(
    pts: np.ndarray,
    cols: np.ndarray,
    focus_pts: np.ndarray,
    highlight_pts: np.ndarray,
    out_dir: Path,
    views: Sequence[Tuple[str, float, float]],
    width: int,
    height: int,
    point_size: float,
    file_prefix: str,
    anchor_name: str,
    highlight_name: str,
    color_anchor: np.ndarray,
    color_highlight: np.ndarray,
) -> Tuple[List[Path], str]:
    out_dir.mkdir(parents=True, exist_ok=True)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(cols)

    center = focus_pts.mean(axis=0)
    extent = float(np.linalg.norm(focus_pts.max(0) - focus_pts.min(0)))
    radius = max(extent * 1.65, 0.12)
    axis_scale = max(extent * 0.70, 0.08)

    renderer = rendering.OffscreenRenderer(width, height)
    sc = renderer.scene
    sc.set_background([0.12, 0.13, 0.15, 1.0])
    sc.scene.set_sun_light([0.35, 0.25, -0.9], [1.0, 1.0, 1.0], 90000)
    sc.scene.enable_sun_light(True)

    z0 = float(focus_pts[:, 2].min()) - 0.01
    ground = o3d.geometry.TriangleMesh.create_box(
        width=extent * 3.0, height=extent * 3.0, depth=0.002
    )
    ground.translate([-1.5 * extent + center[0], -1.5 * extent + center[1], z0])
    ground.paint_uniform_color([0.22, 0.23, 0.25])
    ground.compute_vertex_normals()
    gmat = rendering.MaterialRecord()
    gmat.shader = "defaultLit"
    gmat.base_color = [0.22, 0.23, 0.25, 1.0]
    gmat.base_roughness = 0.9
    sc.add_geometry("ground", ground, gmat)

    amat = rendering.MaterialRecord()
    amat.shader = "defaultLit"
    amat.base_color = [COLOR_AXIS[0], COLOR_AXIS[1], COLOR_AXIS[2], 1.0]
    amat.base_roughness = 0.35
    for i, arrow in enumerate(_axis_group(axis_scale)):
        sc.add_geometry(f"axis_{i}", arrow, amat)

    pc_mat = rendering.MaterialRecord()
    pc_mat.shader = "defaultUnlit"
    pc_mat.point_size = float(point_size)
    sc.add_geometry("pc", pcd, pc_mat)

    scored: List[Tuple[float, str, float, float]] = []
    for name, elev, azim in views:
        eye = _sphere_eye(center, radius, elev, azim)
        scored.append((_view_score(eye, focus_pts, highlight_pts), name, elev, azim))
    scored.sort(reverse=True)
    best_name = scored[0][1]

    saved: List[Path] = []
    up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    fov = 50.0
    for name, elev, azim in views:
        eye = _sphere_eye(center, radius, elev, azim)
        look = center - eye
        look = look / (np.linalg.norm(look) + 1e-9)
        cam_up = up.copy()
        if abs(np.dot(look, up)) > 0.92:
            cam_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        renderer.setup_camera(fov, center, eye, cam_up)
        img = renderer.render_to_image()
        tag = "_RECOMMENDED" if name == best_name else ""
        path = out_dir / f"{file_prefix}_{name}_e{int(elev)}_a{int(azim)}{tag}.png"
        o3d.io.write_image(str(path), img)
        saved.append(path)
        print(f"[saved] {path}{'  ← best' if name == best_name else ''}")

    try:
        import matplotlib.pyplot as plt
        from PIL import Image

        n = len(saved)
        cols_n = min(4, n)
        rows = int(np.ceil(n / cols_n))
        fig, axes = plt.subplots(rows, cols_n, figsize=(3.4 * cols_n, 2.8 * rows))
        axes = np.atleast_1d(axes).ravel()
        for ax in axes:
            ax.axis("off")
        for ax, p in zip(axes, saved):
            ax.imshow(Image.open(p))
            title = p.stem.replace(f"{file_prefix}_", "")
            ax.set_title(title, fontsize=7, color="#eee")
            ax.axis("off")
        fig.patch.set_facecolor("#1a1a1a")
        sheet = out_dir / "00_all_views_contact_sheet.png"
        fig.tight_layout(pad=0.25)
        fig.savefig(sheet, dpi=280, facecolor="#1a1a1a", bbox_inches="tight")
        plt.close(fig)
        print(f"[saved] {sheet}")
        saved.insert(0, sheet)
    except Exception as e:
        print(f"[warn] contact sheet: {e}")

    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8.5, 1.15))
        ax.set_xlim(0, 3.2)
        ax.set_ylim(0, 1)
        ax.axis("off")
        fig.patch.set_facecolor("#1a1a1a")
        items = [
            (color_anchor, f"anchor {_short(anchor_name)}"),
            (color_highlight, f"neighbor {_short(highlight_name)}"),
            (COLOR_OTHER, "other links (gray)"),
        ]
        for i, (c, lab) in enumerate(items):
            ax.add_patch(plt.Rectangle((i * 1.05 + 0.08, 0.38), 0.22, 0.38, color=c))
            ax.text(i * 1.05 + 0.35, 0.55, lab, va="center", fontsize=8.5, color="white")
        ax.text(
            0.08,
            0.12,
            f"Longer red arrows = {_short(anchor_name)} local XYZ",
            fontsize=8,
            color="#ff6666",
        )
        leg = out_dir / "00_legend.png"
        fig.savefig(leg, dpi=200, facecolor="#1a1a1a", bbox_inches="tight")
        plt.close(fig)
        print(f"[saved] {leg}")
    except Exception as e:
        print(f"[warn] legend: {e}")

    del renderer
    return saved, best_name

def export_one_group(
    gen: RobotPointCloudGenerator,
    qpos: np.ndarray,
    anchor: str,
    highlight: str,
    out_dir: Path,
    points_per_link: int,
    point_size: float,
    width: int,
    height: int,
    color_anchor: Optional[np.ndarray] = None,
    color_highlight: Optional[np.ndarray] = None,
) -> str:
    color_anchor = (
        np.asarray(color_anchor, dtype=np.float64)
        if color_anchor is not None
        else COLOR_ANCHOR_DEFAULT.copy()
    )
    color_highlight = (
        np.asarray(color_highlight, dtype=np.float64)
        if color_highlight is not None
        else COLOR_HIGHLIGHT_DEFAULT.copy()
    )

    clouds_w, fk = _world_clouds(gen, qpos)
    if anchor not in fk:
        raise SystemExit(f"anchor {anchor} not in FK: {list(fk)}")
    if highlight not in clouds_w:
        raise SystemExit(f"highlight {highlight} not in clouds: {list(clouds_w)}")

    clouds_a = _to_anchor_frame(clouds_w, fk[anchor])
    pts, cols, focus, highlight_pts = _colorize(
        clouds_a, anchor, highlight, color_anchor, color_highlight
    )

    prefix = f"pc_{_short(anchor)}frame"
    print("=" * 60)
    print("anchor:", anchor, "color=", np.round(color_anchor, 3).tolist())
    print("highlight:", highlight, "color=", np.round(color_highlight, 3).tolist())
    print("qpos left:", np.round(qpos[:7], 3).tolist())
    print(f"points: {len(pts)}  focus: {len(focus)}  highlight: {len(highlight_pts)}")
    print("out_dir:", out_dir)

    paths, best = render_views(
        pts,
        cols,
        focus,
        highlight_pts,
        out_dir,
        DEFAULT_VIEWS,
        width=width,
        height=height,
        point_size=point_size,
        file_prefix=prefix,
        anchor_name=anchor,
        highlight_name=highlight,
        color_anchor=color_anchor,
        color_highlight=color_highlight,
    )

    readme = out_dir / "README.md"
    readme.write_text(
        "\n".join(
            [
                f"# Point cloud in `{anchor}` local frame",
                "",
                f"- **Anchor / camera frame**: `{anchor}`",
                f"- **Anchor color**: `{anchor}` RGB={np.round(color_anchor, 3).tolist()}",
                f"- **Neighbor color**: `{highlight}` RGB={np.round(color_highlight, 3).tolist()}",
                "- **Gray**: other links",
                f"- **Longer red arrows**: local XYZ at `{anchor}` origin",
                f"- Auto-recommended view tagged `*_RECOMMENDED.png` → **{best}**",
                f"- points_per_link={points_per_link}, point_size={point_size}",
                "",
                "Same style as `robot_pc_link6_frame` (colors vary per group). "
                "Generated by `export_robot_pc_link_frame_views.py`.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"Done → {out_dir}  recommended={best}  files={len(paths)}")
    return best

def main():
    parser = argparse.ArgumentParser()
    figs = Path(__file__).resolve().parents[1] / "docs" / "figures"
    parser.add_argument("--out_dir", type=str, default="")
    parser.add_argument("--points_per_link", type=int, default=400)
    parser.add_argument("--point_size", type=float, default=12.0)
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=1200)
    parser.add_argument("--anchor", type=str, default="fl_link6")
    parser.add_argument("--highlight", type=str, default="fl_link7")
    parser.add_argument(
        "--batch_three",
        action="store_true",
        help="Export 3 groups: link1/link2, link3/link4, link7/link6 (same style as link6)",
    )
    args = parser.parse_args()

    gen = RobotPointCloudGenerator(
        points_per_link=args.points_per_link,
        sample_region="surface",
        device="cpu",
    )
    qpos = build_show_qpos()

    if args.batch_three:
        jobs = [
            (a, h, ca, ch) for (a, h, ca, ch) in BATCH_THREE
        ]
    else:
        jobs = [
            (args.anchor, args.highlight, COLOR_ANCHOR_DEFAULT, COLOR_HIGHLIGHT_DEFAULT)
        ]

    for anchor, highlight, color_a, color_h in jobs:
        if args.out_dir and not args.batch_three:
            out_dir = Path(args.out_dir)
        else:
            out_dir = figs / f"robot_pc_{_short(anchor)}_frame"
        export_one_group(
            gen,
            qpos,
            anchor,
            highlight,
            out_dir,
            points_per_link=args.points_per_link,
            point_size=args.point_size,
            width=args.width,
            height=args.height,
            color_anchor=color_a,
            color_highlight=color_h,
        )

if __name__ == "__main__":
    main()
