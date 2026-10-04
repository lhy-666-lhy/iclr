#!/usr/bin/env python3
"""Export left-arm colored geometry PCs on a table in a simple lab scene.

Sparse large points, per-link colors, articulated pose, multi-view.

Usage:
  python policy/SPACE/scripts/export_robot_pc_views.py \\
      --out_dir policy/SPACE/docs/figures/robot_pc_views \\
      --points_per_link 400 --point_size 12
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

os.environ.setdefault("EGL_PLATFORM", "surfaceless")

import open3d as o3d
from open3d.visualization import rendering

BLOCK_DIR = Path(__file__).resolve().parents[1] / "block"
if str(BLOCK_DIR) not in sys.path:
    sys.path.insert(0, str(BLOCK_DIR))

from make_robot import RobotPointCloudGenerator, _transform_local_points  # noqa: E402

# High-contrast per-link colors (still vivid against a normal room)
LINK_COLORS = np.array(
    [
        [1.000, 0.200, 0.200],  # base  vivid red
        [1.000, 0.650, 0.000],  # 1     orange
        [0.150, 0.750, 1.000],  # 2     cyan-blue
        [0.200, 1.000, 0.300],  # 3     lime green
        [0.850, 0.350, 1.000],  # 4     violet
        [1.000, 0.850, 0.150],  # 5     gold
        [1.000, 0.400, 0.850],  # 6     hot pink
        [0.980, 0.980, 0.980],  # 7     near-white
        [0.950, 1.000, 0.200],  # 8     yellow
    ],
    dtype=np.float64,
)

# Prefer views that show the table under the arm
DEFAULT_VIEWS: List[Tuple[str, float, float]] = [
    ("front", 22.0, 5.0),
    ("front_left", 24.0, 40.0),
    ("front_right", 24.0, -40.0),
    ("side_left", 18.0, 88.0),
    ("side_right", 18.0, -88.0),
    ("top", 68.0, 30.0),
    ("three_quarter_L", 28.0, 48.0),
    ("three_quarter_R", 28.0, -48.0),
    ("back", 22.0, 175.0),
    ("low_front", 12.0, 28.0),
]

def gen_idx(name: str) -> int:
    if "base" in name:
        return 0
    for i in range(1, 9):
        if name.endswith(f"link{i}") or name.split("_")[-1] == f"link{i}":
            return i
    return 0

def build_show_qpos() -> np.ndarray:
    """Articulated left-arm pose that clearly shows the full chain (not zero)."""
    q = np.zeros(14, dtype=np.float32)
    q[0] = 0.45
    q[1] = 0.70
    q[2] = 1.05
    q[3] = -0.95
    q[4] = 0.70
    q[5] = 0.40
    q[6] = 0.75
    return q

def left_geometry_world(
    gen: RobotPointCloudGenerator, qpos: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, List[str], Dict[str, np.ndarray]]:
    """Return pts, colors, names, and per-link world clouds."""
    q = torch.as_tensor(qpos, dtype=torch.float32).unsqueeze(0)
    fk = gen.forward_kinematics(q, arm="left")
    pts_list: List[np.ndarray] = []
    col_list: List[np.ndarray] = []
    names: List[str] = []
    by_link: Dict[str, np.ndarray] = {}
    for name, pts_local in zip(gen.left_links, gen.left_local):
        if name not in fk["left"]:
            continue
        pw = _transform_local_points(fk["left"][name], pts_local)[0]
        pw = pw.detach().cpu().numpy().astype(np.float64)
        c = LINK_COLORS[gen_idx(name) % len(LINK_COLORS)]
        pts_list.append(pw)
        col_list.append(np.tile(c, (len(pw), 1)))
        names.append(name)
        by_link[name] = pw
    pts = np.concatenate(pts_list, axis=0)
    cols = np.concatenate(col_list, axis=0)
    return pts, cols, names, by_link

def _box(center: np.ndarray, size_xyz: np.ndarray, color: Sequence[float]) -> o3d.geometry.TriangleMesh:
    """Axis-aligned box centered at ``center`` with full extents ``size_xyz``."""
    mesh = o3d.geometry.TriangleMesh.create_box(
        width=float(size_xyz[0]),
        height=float(size_xyz[1]),
        depth=float(size_xyz[2]),
    )
    mesh.translate(center - 0.5 * np.asarray(size_xyz, dtype=np.float64))
    mesh.paint_uniform_color(list(color))
    mesh.compute_vertex_normals()
    return mesh

def build_lab_scene(
    pts: np.ndarray, by_link: Dict[str, np.ndarray]
) -> List[Tuple[str, o3d.geometry.TriangleMesh, rendering.MaterialRecord]]:
    """Floor + walls + wooden table under the arm base."""
    # Prefer base link for seating; fall back to global min-z
    base_name = next((n for n in by_link if "base" in n), None)
    seat = by_link[base_name] if base_name is not None else pts
    seat_xy = seat[:, :2].mean(axis=0)
    table_top_z = float(seat[:, 2].min()) - 0.002

    # Table sized relative to arm span
    arm_span = float(np.linalg.norm(pts.max(0) - pts.min(0)))
    table_w = max(1.10, arm_span * 1.35)  # X
    table_d = max(0.70, arm_span * 0.95)  # Y
    top_t = 0.04
    leg_h = 0.72
    leg_t = 0.05

    table_cx, table_cy = float(seat_xy[0]), float(seat_xy[1])
    # Shift table a bit so arm sits near center-front of table
    table_cy = table_cy + 0.05

    wood = [0.62, 0.45, 0.28]
    wood_dark = [0.42, 0.28, 0.16]
    floor_c = [0.55, 0.55, 0.52]
    wall_c = [0.82, 0.84, 0.86]
    wall_accent = [0.74, 0.78, 0.82]

    items: List[Tuple[str, o3d.geometry.TriangleMesh, rendering.MaterialRecord]] = []

    def lit(color_rgb, metallic=0.12, roughness=0.75) -> rendering.MaterialRecord:
        m = rendering.MaterialRecord()
        m.shader = "defaultLit"
        m.base_color = [color_rgb[0], color_rgb[1], color_rgb[2], 1.0]
        m.base_roughness = roughness
        m.base_metallic = metallic
        return m

    # Floor
    floor_z = table_top_z - leg_h - top_t
    floor = _box(
        np.array([table_cx, table_cy, floor_z - 0.02]),
        np.array([4.0, 4.0, 0.04]),
        floor_c,
    )
    items.append(("floor", floor, lit(floor_c, roughness=0.9)))

    # Back + side walls (room)
    wall_h = 2.2
    wall_z = floor_z + wall_h * 0.5
    back = _box(
        np.array([table_cx, table_cy + 1.85, wall_z]),
        np.array([4.0, 0.06, wall_h]),
        wall_c,
    )
    items.append(("wall_back", back, lit(wall_c, roughness=0.85)))
    side = _box(
        np.array([table_cx - 1.85, table_cy, wall_z]),
        np.array([0.06, 4.0, wall_h]),
        wall_accent,
    )
    items.append(("wall_side", side, lit(wall_accent, roughness=0.85)))

    # Table top
    top_center = np.array([table_cx, table_cy, table_top_z - 0.5 * top_t])
    top = _box(top_center, np.array([table_w, table_d, top_t]), wood)
    items.append(("table_top", top, lit(wood, roughness=0.55)))

    # Four legs
    inset_x = 0.5 * table_w - 0.06
    inset_y = 0.5 * table_d - 0.06
    leg_z = table_top_z - top_t - 0.5 * leg_h
    for i, (sx, sy) in enumerate(
        [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    ):
        leg = _box(
            np.array([table_cx + sx * inset_x, table_cy + sy * inset_y, leg_z]),
            np.array([leg_t, leg_t, leg_h]),
            wood_dark,
        )
        items.append((f"leg_{i}", leg, lit(wood_dark, roughness=0.7)))

    # Small mat under base for “mounted on table” cue
    mat_c = [0.18, 0.18, 0.20]
    mat = _box(
        np.array([float(seat_xy[0]), float(seat_xy[1]), table_top_z + 0.004]),
        np.array([0.22, 0.22, 0.008]),
        mat_c,
    )
    items.append(("base_mat", mat, lit(mat_c, roughness=0.6)))

    return items

def _sphere_eye(center: np.ndarray, radius: float, elev_deg: float, azim_deg: float) -> np.ndarray:
    elev = np.deg2rad(elev_deg)
    azim = np.deg2rad(azim_deg)
    x = radius * np.cos(elev) * np.cos(azim)
    y = radius * np.cos(elev) * np.sin(azim)
    z = radius * np.sin(elev)
    return center + np.array([x, y, z], dtype=np.float64)

def render_pc_views(
    pts: np.ndarray,
    cols: np.ndarray,
    by_link: Dict[str, np.ndarray],
    out_dir: Path,
    views: Sequence[Tuple[str, float, float]],
    width: int = 1600,
    height: int = 1200,
    point_size: float = 12.0,
) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(cols)

    scene_items = build_lab_scene(pts, by_link)

    # Frame arm + a bit of table
    center = pts.mean(axis=0).copy()
    center[2] = 0.55 * pts[:, 2].mean() + 0.45 * pts[:, 2].min()
    extent = float(np.linalg.norm(pts.max(0) - pts.min(0)))
    radius = max(extent * 1.55, 0.85)

    renderer = rendering.OffscreenRenderer(width, height)
    sc = renderer.scene
    # Soft lab / sky backdrop
    sc.set_background([0.70, 0.76, 0.84, 1.0])
    sc.scene.set_sun_light([0.45, 0.20, -0.85], [1.0, 0.98, 0.94], 95000)
    sc.scene.enable_sun_light(True)
    # Gentle fill so table/walls are readable
    try:
        sc.scene.set_indirect_light_intensity(25000)
    except Exception:
        pass

    for name, mesh, mat in scene_items:
        sc.add_geometry(name, mesh, mat)

    pc_mat = rendering.MaterialRecord()
    pc_mat.shader = "defaultUnlit"
    pc_mat.point_size = float(point_size)
    sc.add_geometry("arm_pc", pcd, pc_mat)

    saved: List[Path] = []
    up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    fov = 48.0
    for name, elev, azim in views:
        eye = _sphere_eye(center, radius, elev, azim)
        # Keep camera a bit above floor
        eye[2] = max(eye[2], center[2] - 0.15)
        look = center - eye
        look = look / (np.linalg.norm(look) + 1e-9)
        cam_up = up.copy()
        if abs(np.dot(look, up)) > 0.92:
            cam_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        renderer.setup_camera(fov, center, eye, cam_up)
        img = renderer.render_to_image()
        path = out_dir / f"pc_{name}_e{int(elev)}_a{int(azim)}.png"
        o3d.io.write_image(str(path), img)
        saved.append(path)
        print(f"[saved] {path}")

    # Contact sheet
    try:
        import matplotlib.pyplot as plt
        from PIL import Image

        n = len(saved)
        cols_n = min(5, n)
        rows = int(np.ceil(n / cols_n))
        fig, axes = plt.subplots(rows, cols_n, figsize=(3.2 * cols_n, 2.6 * rows))
        axes = np.atleast_1d(axes).ravel()
        for ax in axes:
            ax.axis("off")
            ax.set_facecolor("#f2f2f2")
        for ax, p in zip(axes, saved):
            ax.imshow(Image.open(p))
            ax.set_title(p.stem.replace("pc_", ""), fontsize=8, color="#222222")
            ax.axis("off")
        fig.patch.set_facecolor("white")
        sheet = out_dir / "00_all_views_contact_sheet.png"
        fig.tight_layout(pad=0.3)
        fig.savefig(sheet, dpi=300, facecolor="white", bbox_inches="tight")
        plt.close(fig)
        print(f"[saved] {sheet}")
        saved.insert(0, sheet)
    except Exception as e:
        print(f"[warn] contact sheet skipped: {e}")

    # Legend
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8.5, 1.1))
        ax.set_xlim(0, len(LINK_COLORS))
        ax.set_ylim(0, 1)
        ax.axis("off")
        fig.patch.set_facecolor("white")
        labels = ["base", "link1", "link2", "link3", "link4", "link5", "link6", "link7", "link8"]
        for i, (c, lab) in enumerate(zip(LINK_COLORS, labels)):
            ax.add_patch(plt.Rectangle((i + 0.1, 0.25), 0.8, 0.5, color=c))
            ax.text(i + 0.5, 0.05, lab, ha="center", va="bottom", fontsize=9, color="#222")
        legend = out_dir / "00_link_color_legend.png"
        fig.savefig(legend, dpi=200, facecolor="white", bbox_inches="tight")
        plt.close(fig)
        print(f"[saved] {legend}")
    except Exception as e:
        print(f"[warn] legend skipped: {e}")

    del renderer
    return saved

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out_dir",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "docs" / "figures" / "robot_pc_views"),
    )
    parser.add_argument("--points_per_link", type=int, default=400)
    parser.add_argument("--point_size", type=float, default=12.0)
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=1200)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    gen = RobotPointCloudGenerator(
        points_per_link=args.points_per_link,
        sample_region="surface",
        device="cpu",
    )
    qpos = build_show_qpos()
    print("left links:", gen.left_links)
    print("points_per_link:", args.points_per_link)
    print("qpos (left):", np.round(qpos[:7], 3).tolist())

    pts, cols, names, by_link = left_geometry_world(gen, qpos)
    print(f"total points: {len(pts)}  links: {names}")

    paths = render_pc_views(
        pts,
        cols,
        by_link,
        out_dir,
        DEFAULT_VIEWS,
        width=args.width,
        height=args.height,
        point_size=args.point_size,
    )

    readme = out_dir / "README.md"
    lines = [
        "# Left-arm colored geometry point cloud views",
        "",
        "Per-link colors · **lab table scene** (floor/walls/table) · sparse large points · articulated pose.",
        "",
        f"- points_per_link = **{args.points_per_link}** (total ≈ {len(pts)})",
        f"- point_size = **{args.point_size}**",
        f"- left qpos = `{np.round(qpos[:7], 3).tolist()}`",
        "",
        "| File | View |",
        "|------|------|",
    ]
    for name, elev, azim in DEFAULT_VIEWS:
        lines.append(f"| `pc_{name}_e{int(elev)}_a{int(azim)}.png` | elev={elev}, azim={azim} |")
    lines += [
        "",
        "Also: `00_all_views_contact_sheet.png`, `00_link_color_legend.png`.",
        "",
        "Generated by `export_robot_pc_views.py`.",
    ]
    readme.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Done → {out_dir} ({len(paths)} files)")

if __name__ == "__main__":
    main()
