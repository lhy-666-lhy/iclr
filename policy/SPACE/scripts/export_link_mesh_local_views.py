#!/usr/bin/env python3
"""Export per-link collision meshes in each link's own frame + surface PC cubes.

Style:
  - one figure per link (only that link)
  - very transparent light-gray mesh + faint black edges (shape ghost)
  - colored surface point cloud as small cubes (blue / green / purple / orange)
  - red arrows = link-local XYZ at the origin (unchanged)

Usage:
  python policy/SPACE/scripts/export_link_mesh_local_views.py \\
      --out_dir policy/SPACE/docs/figures/robot_link_mesh_local \\
      --links fl_link1,fl_link3,fl_link6,fl_link7 \\
      --n_points 96
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

os.environ.setdefault("EGL_PLATFORM", "surfaceless")

import open3d as o3d
from open3d.visualization import rendering

BLOCK_DIR = Path(__file__).resolve().parents[1] / "block"
if str(BLOCK_DIR) not in sys.path:
    sys.path.insert(0, str(BLOCK_DIR))

from make_robot import (  # noqa: E402
    RobotPointCloudGenerator,
    _discover_arm_links,
    _preferred_arm_link_order,
    _sample_mesh_points,
    _stable_link_seed,
)

COLOR_MESH = np.array([0.78, 0.78, 0.80], dtype=np.float64)
COLOR_EDGE = np.array([0.12, 0.12, 0.12], dtype=np.float64)
COLOR_AXIS = np.array([1.00, 0.08, 0.08], dtype=np.float64)

# Per-link PC colors: blue, green, purple, orange
PC_COLORS: Tuple[np.ndarray, ...] = (
    np.array([0.15, 0.45, 1.00], dtype=np.float64),  # blue
    np.array([0.15, 0.72, 0.28], dtype=np.float64),  # green
    np.array([0.62, 0.28, 0.82], dtype=np.float64),  # purple
    np.array([1.00, 0.55, 0.08], dtype=np.float64),  # orange
)
PC_COLOR_NAMES = ("blue", "green", "purple", "orange")

MESH_ALPHA = 0.10
EDGE_ALPHA = 0.12

DEFAULT_LINKS = ("fl_link1", "fl_link3", "fl_link6", "fl_link7")
DEFAULT_VIEW = ("three_quarter", 22.0, 45.0)

def _short(name: str) -> str:
    return name.split("_", 1)[-1] if "_" in name else name

def _load_left_meshes(gen: RobotPointCloudGenerator) -> Dict[str, object]:
    names, meshes = _discover_arm_links(gen.urdf_path, gen.left_prefix)
    ordered = _preferred_arm_link_order(names, gen.left_prefix)
    mmap = dict(zip(names, meshes))
    return {n: mmap[n] for n in ordered if n in mmap and mmap[n] is not None}

def _trimesh_to_o3d(mesh) -> o3d.geometry.TriangleMesh:
    tm = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.asarray(mesh.vertices, dtype=np.float64)),
        o3d.utility.Vector3iVector(np.asarray(mesh.faces, dtype=np.int32)),
    )
    tm.paint_uniform_color(COLOR_MESH.tolist())
    tm.compute_vertex_normals()
    return tm

def _mesh_edges(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.LineSet:
    tris = np.asarray(mesh.triangles, dtype=np.int64)
    if len(tris) == 0:
        return o3d.geometry.LineSet()
    edges = np.vstack([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    edges = np.sort(edges, axis=1)
    edges = np.unique(edges, axis=0)
    ls = o3d.geometry.LineSet(
        points=mesh.vertices,
        lines=o3d.utility.Vector2iVector(edges.astype(np.int32)),
    )
    ls.paint_uniform_color(COLOR_EDGE.tolist())
    return ls

def _axis_arrow(axis: str, length: float, radius: float) -> o3d.geometry.TriangleMesh:
    cone_h = length * 0.20
    cyl_h = length - cone_h
    arrow = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=radius,
        cone_radius=radius * 2.4,
        cylinder_height=cyl_h,
        cone_height=cone_h,
        resolution=24,
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

def _axis_group(extent: float) -> List[o3d.geometry.TriangleMesh]:
    L = max(extent * 0.55, 0.035)
    r = max(extent * 0.018, 0.0012)
    return [_axis_arrow("x", L, r), _axis_arrow("y", L, r), _axis_arrow("z", L, r)]

def _sample_local_pc(mesh_tm, link_name: str, n_points: int) -> np.ndarray:
    pts = _sample_mesh_points(
        mesh_tm,
        int(n_points),
        seed=_stable_link_seed(link_name),
        region="surface",
    )
    return np.asarray(pts, dtype=np.float64)

def _pc_cubes(pts: np.ndarray, color: np.ndarray, cube_size: float) -> o3d.geometry.TriangleMesh:
    """Merge axis-aligned cubes centered at each point (small square markers)."""
    half = float(cube_size) * 0.5
    merged = o3d.geometry.TriangleMesh()
    for p in pts:
        box = o3d.geometry.TriangleMesh.create_box(
            width=cube_size, height=cube_size, depth=cube_size
        )
        box.translate(p - half)
        merged += box
    if len(pts) > 0:
        merged.paint_uniform_color(color.tolist())
        merged.compute_vertex_normals()
    return merged

def _sphere_eye(center: np.ndarray, radius: float, elev_deg: float, azim_deg: float) -> np.ndarray:
    elev = np.deg2rad(elev_deg)
    azim = np.deg2rad(azim_deg)
    x = radius * np.cos(elev) * np.cos(azim)
    y = radius * np.cos(elev) * np.sin(azim)
    z = radius * np.sin(elev)
    return center + np.array([x, y, z], dtype=np.float64)

def render_one_link(
    mesh_tm,
    link_name: str,
    pc_color: np.ndarray,
    out_path: Path,
    elev: float,
    azim: float,
    n_points: int,
    width: int = 1400,
    height: int = 1100,
) -> Path:
    o3d_mesh = _trimesh_to_o3d(mesh_tm)
    edges = _mesh_edges(o3d_mesh)

    aabb = o3d_mesh.get_axis_aligned_bounding_box()
    mn = np.asarray(aabb.min_bound, dtype=np.float64)
    mx = np.asarray(aabb.max_bound, dtype=np.float64)
    mn = np.minimum(mn, np.zeros(3))
    mx = np.maximum(mx, np.zeros(3))
    center = 0.5 * (mn + mx)
    extent = float(np.linalg.norm(mx - mn))
    radius = max(extent * 1.35, 0.08)

    pts = _sample_local_pc(mesh_tm, link_name, n_points)
    cube_size = max(extent * 0.028, 0.0028)
    cubes = _pc_cubes(pts, pc_color, cube_size)

    renderer = rendering.OffscreenRenderer(width, height)
    scene = renderer.scene
    scene.set_background([1.0, 1.0, 1.0, 1.0])
    scene.scene.set_sun_light([0.35, -0.25, -0.9], [1.0, 1.0, 1.0], 75000)
    scene.scene.enable_sun_light(True)

    # Ghost mesh (very transparent)
    mat_mesh = rendering.MaterialRecord()
    mat_mesh.shader = "defaultLitTransparency"
    mat_mesh.base_color = [COLOR_MESH[0], COLOR_MESH[1], COLOR_MESH[2], MESH_ALPHA]
    mat_mesh.base_roughness = 0.85
    scene.add_geometry("mesh", o3d_mesh, mat_mesh)

    mat_edge = rendering.MaterialRecord()
    mat_edge.shader = "unlitLine"
    mat_edge.line_width = 0.9
    mat_edge.base_color = [COLOR_EDGE[0], COLOR_EDGE[1], COLOR_EDGE[2], EDGE_ALPHA]
    scene.add_geometry("edges", edges, mat_edge)

    # Opaque colored cubes (surface PC)
    mat_pc = rendering.MaterialRecord()
    mat_pc.shader = "defaultLit"
    mat_pc.base_color = [pc_color[0], pc_color[1], pc_color[2], 1.0]
    scene.add_geometry("pc_cubes", cubes, mat_pc)

    # Red axes unchanged
    mat_axis = rendering.MaterialRecord()
    mat_axis.shader = "defaultLit"
    for i, arrow in enumerate(_axis_group(extent)):
        scene.add_geometry(f"axis_{i}", arrow, mat_axis)

    up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    eye = _sphere_eye(center, radius, elev, azim)
    look = center - eye
    look = look / (np.linalg.norm(look) + 1e-9)
    cam_up = up.copy()
    if abs(np.dot(look, up)) > 0.92:
        cam_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    renderer.setup_camera(40.0, center, eye, cam_up)

    img = renderer.render_to_image()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_image(str(out_path), img)
    del renderer
    print(f"[saved] {out_path}  (n_points={len(pts)}, cube={cube_size:.4f})")
    return out_path

def write_contact_sheet(paths: Sequence[Path], out_path: Path, titles: Sequence[str], dpi: int = 300) -> None:
    try:
        import matplotlib.pyplot as plt
        from PIL import Image
    except Exception as e:
        print(f"[warn] contact sheet skipped: {e}")
        return

    n = len(paths)
    cols = min(2, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 4.2 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, p, t in zip(axes, paths, titles):
        ax.imshow(Image.open(p))
        ax.set_title(t, fontsize=11)
        ax.axis("off")
    fig.patch.set_facecolor("white")
    fig.tight_layout(pad=0.4)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_path}")

def parse_links(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out_dir",
        type=str,
        default=str(
            Path(__file__).resolve().parents[1] / "docs" / "figures" / "robot_link_mesh_local"
        ),
    )
    parser.add_argument(
        "--links",
        type=str,
        default=",".join(DEFAULT_LINKS),
        help="Comma-separated link names (default: 4 diverse left-arm links)",
    )
    parser.add_argument("--n_points", type=int, default=96, help="Surface samples per link")
    parser.add_argument("--elev", type=float, default=DEFAULT_VIEW[1])
    parser.add_argument("--azim", type=float, default=DEFAULT_VIEW[2])
    parser.add_argument("--width", type=int, default=1400)
    parser.add_argument("--height", type=int, default=1100)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    link_names = parse_links(args.links)

    gen = RobotPointCloudGenerator(points_per_link=4, sample_region="surface", device="cpu")
    mesh_map = _load_left_meshes(gen)
    print("available left links:", list(mesh_map.keys()))

    missing = [n for n in link_names if n not in mesh_map]
    if missing:
        raise SystemExit(f"links not found: {missing}\navailable: {list(mesh_map.keys())}")

    saved: List[Path] = []
    titles: List[str] = []
    for i, name in enumerate(link_names):
        color = PC_COLORS[i % len(PC_COLORS)]
        cname = PC_COLOR_NAMES[i % len(PC_COLOR_NAMES)]
        short = _short(name)
        path = out_dir / f"mesh_local_{short}_e{int(args.elev)}_a{int(args.azim)}.png"
        render_one_link(
            mesh_map[name],
            link_name=name,
            pc_color=color,
            out_path=path,
            elev=args.elev,
            azim=args.azim,
            n_points=args.n_points,
            width=args.width,
            height=args.height,
        )
        saved.append(path)
        titles.append(f"{name} · {cname} PC")

    sheet = out_dir / "00_four_links_contact_sheet.png"
    write_contact_sheet(saved, sheet, titles, dpi=args.dpi)

    readme = out_dir / "README.md"
    lines = [
        "# Per-link mesh + surface PC in own link frame",
        "",
        "Very transparent gray mesh / faint edges · colored **cube** PC · **red arrows** = link XYZ.",
        "",
        f"Generated by `export_link_mesh_local_views.py` "
        f"(n_points={args.n_points}, elev={args.elev}, azim={args.azim}).",
        "",
        "| File | Link | PC color |",
        "|------|------|----------|",
    ]
    for i, (name, p) in enumerate(zip(link_names, saved)):
        cname = PC_COLOR_NAMES[i % len(PC_COLOR_NAMES)]
        lines.append(f"| `{p.name}` | `{name}` | {cname} |")
    lines.append("")
    lines.append("Also: `00_four_links_contact_sheet.png`.")
    readme.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Done → {out_dir}")

if __name__ == "__main__":
    main()
