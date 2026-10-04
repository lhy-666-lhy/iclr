#!/usr/bin/env python3
"""Export full-arm ghost mesh + per-link colored point clouds + red link axes.

Style:
  - almost-transparent gray mesh fill (no edge lines)
  - each link: distinct-color surface PC (small cubes)
  - red XYZ arrows at every link origin
  - long green XYZ arrows at world origin = midpoint of fl/fr base_link centers
  - ground on the world XY plane (same plane as green X/Y arrows)

Usage:
  python policy/SPACE/scripts/export_full_arm_mesh_axes.py \\
      --out_dir policy/SPACE/docs/figures/robot_full_mesh_axes \\
      --arms both --n_points 64
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

SCRIPTS_DIR = Path(__file__).resolve().parent
BLOCK_DIR = Path(__file__).resolve().parents[1] / "block"
for p in (str(SCRIPTS_DIR), str(BLOCK_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from make_robot import (  # noqa: E402
    RobotPointCloudGenerator,
    _transform_local_points,
)
from export_robot_mesh_views import (  # noqa: E402
    assemble_world_meshes,
    build_demo_qpos,
)

COLOR_MESH = np.array([0.78, 0.78, 0.80], dtype=np.float64)
COLOR_AXIS = np.array([1.00, 0.08, 0.08], dtype=np.float64)  # link frames
COLOR_WORLD = np.array([0.05, 0.78, 0.20], dtype=np.float64)  # world frame (green)
COLOR_GROUND = np.array([0.82, 0.82, 0.80], dtype=np.float64)
COLOR_GROUND_GRID = np.array([0.62, 0.62, 0.60], dtype=np.float64)

MESH_ALPHA = 0.05
EDGE_ALPHA = 0.05

DEFAULT_VIEWS: List[Tuple[str, float, float]] = [
    ("three_quarter_L", 22.0, 45.0),
    ("front", 15.0, 0.0),
    ("side_left", 12.0, 90.0),
]

def _link_palette(n: int) -> np.ndarray:
    """Distinct RGB colors for n links (tab20-like, cycled if needed)."""
    base = np.array(
        [
            [0.121, 0.466, 0.705],
            [1.000, 0.498, 0.055],
            [0.172, 0.627, 0.172],
            [0.839, 0.153, 0.157],
            [0.580, 0.404, 0.741],
            [0.549, 0.337, 0.294],
            [0.890, 0.467, 0.761],
            [0.498, 0.498, 0.498],
            [0.737, 0.741, 0.133],
            [0.090, 0.745, 0.812],
            [0.682, 0.780, 0.910],
            [1.000, 0.733, 0.471],
            [0.596, 0.875, 0.541],
            [1.000, 0.596, 0.588],
            [0.773, 0.690, 0.835],
            [0.768, 0.612, 0.580],
            [0.969, 0.714, 0.824],
            [0.780, 0.780, 0.780],
            [0.859, 0.859, 0.553],
            [0.620, 0.855, 0.898],
        ],
        dtype=np.float64,
    )
    return np.stack([base[i % len(base)] for i in range(n)], axis=0)

def _axis_arrow(
    axis: str,
    length: float,
    radius: float,
    color: np.ndarray = COLOR_AXIS,
) -> o3d.geometry.TriangleMesh:
    cone_h = length * 0.26
    cyl_h = max(length - cone_h, length * 0.45)
    arrow = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=radius,
        cone_radius=radius * 2.4,
        cylinder_height=cyl_h,
        cone_height=cone_h,
        resolution=20,
        cylinder_split=4,
        cone_split=1,
    )
    arrow.paint_uniform_color(np.asarray(color, dtype=np.float64).tolist())
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

def _axis_group(
    length: float,
    radius: float,
    color: np.ndarray = COLOR_AXIS,
) -> List[o3d.geometry.TriangleMesh]:
    return [
        _axis_arrow("x", length, radius, color),
        _axis_arrow("y", length, radius, color),
        _axis_arrow("z", length, radius, color),
    ]

def _to_gray(meshes: Sequence[o3d.geometry.TriangleMesh]) -> List[o3d.geometry.TriangleMesh]:
    out: List[o3d.geometry.TriangleMesh] = []
    for m in meshes:
        g = o3d.geometry.TriangleMesh(m)
        g.paint_uniform_color(COLOR_MESH.tolist())
        g.compute_vertex_normals()
        out.append(g)
    return out

def _pc_cubes(pts: np.ndarray, color: np.ndarray, cube_size: float) -> o3d.geometry.TriangleMesh:
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

def _collect_link_Ts(
    gen: RobotPointCloudGenerator, qpos: np.ndarray, arms: str
) -> List[Tuple[str, np.ndarray]]:
    q = torch.as_tensor(qpos, dtype=torch.float32)
    fk = gen.forward_kinematics(q, arm=arms)
    items: List[Tuple[str, np.ndarray]] = []
    arm_items = []
    if arms in ("left", "both"):
        arm_items.append(("left", gen.left_links))
    if arms in ("right", "both"):
        arm_items.append(("right", gen.right_links))
    for arm_key, links in arm_items:
        for name in links:
            if name not in fk[arm_key]:
                continue
            T = fk[arm_key][name][0].detach().cpu().numpy().astype(np.float64)
            items.append((name, T))
    return items

def _world_link_clouds(
    gen: RobotPointCloudGenerator, qpos: np.ndarray, arms: str
) -> List[Tuple[str, np.ndarray]]:
    """Per-link surface PC in world frame."""
    q = torch.as_tensor(qpos, dtype=torch.float32).unsqueeze(0)
    fk = gen.forward_kinematics(q, arm=arms)
    out: List[Tuple[str, np.ndarray]] = []
    arm_packs = []
    if arms in ("left", "both"):
        arm_packs.append(("left", gen.left_links, gen.left_local))
    if arms in ("right", "both"):
        arm_packs.append(("right", gen.right_links, gen.right_local))
    for arm_key, links, locals_list in arm_packs:
        for name, pts_local in zip(links, locals_list):
            if name not in fk[arm_key]:
                continue
            pw = _transform_local_points(fk[arm_key][name], pts_local)[0]
            out.append((name, pw.detach().cpu().numpy().astype(np.float64)))
    return out

def _base_center_midpoint(
    gen: RobotPointCloudGenerator, qpos: np.ndarray
) -> np.ndarray:
    """Midpoint of left/right base_link origins in FK world frame."""
    q = torch.as_tensor(qpos, dtype=torch.float32)
    fk = gen.forward_kinematics(q, arm="both")
    origins: List[np.ndarray] = []
    for arm_key, name in (("left", "fl_base_link"), ("right", "fr_base_link")):
        if arm_key in fk and name in fk[arm_key]:
            T = fk[arm_key][name][0].detach().cpu().numpy().astype(np.float64)
            origins.append(T[:3, 3].copy())
    if not origins:
        return np.zeros(3, dtype=np.float64)
    return np.mean(np.stack(origins, axis=0), axis=0)

def _translate_meshes(
    meshes: Sequence[o3d.geometry.TriangleMesh], delta: np.ndarray
) -> List[o3d.geometry.TriangleMesh]:
    out: List[o3d.geometry.TriangleMesh] = []
    for m in meshes:
        g = o3d.geometry.TriangleMesh(m)
        g.translate(delta)
        out.append(g)
    return out

def _sphere_eye(center: np.ndarray, radius: float, elev_deg: float, azim_deg: float) -> np.ndarray:
    elev = np.deg2rad(elev_deg)
    azim = np.deg2rad(azim_deg)
    x = radius * np.cos(elev) * np.cos(azim)
    y = radius * np.cos(elev) * np.sin(azim)
    z = radius * np.sin(elev)
    return center + np.array([x, y, z], dtype=np.float64)

def _make_ground(
    cx: float,
    cy: float,
    z: float,
    size: float,
    thickness: float = 0.004,
    n_grid: int = 10,
) -> Tuple[o3d.geometry.TriangleMesh, o3d.geometry.LineSet]:
    """Horizontal ground slab + light grid, centered at (cx, cy), top face at z."""
    half = size * 0.5
    ground = o3d.geometry.TriangleMesh.create_box(
        width=size, height=size, depth=thickness
    )
    # create_box sits in [0,size]^3; move so top face is at z and center at (cx,cy)
    ground.translate([cx - half, cy - half, z - thickness])
    ground.paint_uniform_color(COLOR_GROUND.tolist())
    ground.compute_vertex_normals()

    # Grid on top face
    xs = np.linspace(cx - half, cx + half, n_grid + 1)
    ys = np.linspace(cy - half, cy + half, n_grid + 1)
    pts: List[np.ndarray] = []
    lines: List[List[int]] = []
    z_eps = z + 1e-4
    idx = 0
    for y in ys:
        pts.append(np.array([cx - half, y, z_eps]))
        pts.append(np.array([cx + half, y, z_eps]))
        lines.append([idx, idx + 1])
        idx += 2
    for x in xs:
        pts.append(np.array([x, cy - half, z_eps]))
        pts.append(np.array([x, cy + half, z_eps]))
        lines.append([idx, idx + 1])
        idx += 2
    grid = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(np.stack(pts, axis=0)),
        lines=o3d.utility.Vector2iVector(np.asarray(lines, dtype=np.int32)),
    )
    grid.paint_uniform_color(COLOR_GROUND_GRID.tolist())
    return ground, grid

def _save_png(path: Path, img) -> None:
    o3d.io.write_image(str(path), img)
    try:
        from PIL import Image

        Image.open(path).convert("RGB").save(path, format="PNG", optimize=True)
    except Exception as e:
        print(f"[warn] PIL re-encode skipped: {e}")

def render_full_arm_with_axes(
    meshes: Sequence[o3d.geometry.TriangleMesh],
    link_Ts: Sequence[Tuple[str, np.ndarray]],
    link_clouds: Sequence[Tuple[str, np.ndarray]],
    out_dir: Path,
    views: Sequence[Tuple[str, float, float]],
    axis_length: float,
    axis_radius: float,
    world_axis_length: float,
    world_axis_radius: float,
    width: int,
    height: int,
    primary_only: bool,
    mesh_alpha: float,
    edge_alpha: float,
    cube_scale: float,
    world_origin: np.ndarray,
) -> List[Path]:
    """Render with world origin = base-center midpoint (scene translated by -origin)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    origin = np.asarray(world_origin, dtype=np.float64).reshape(3)
    shift = -origin  # so base midpoint becomes (0,0,0)

    # Shift meshes / frames / clouds into base-centered world
    meshes_c = _translate_meshes(meshes, shift)
    link_Ts_c: List[Tuple[str, np.ndarray]] = []
    for name, T in link_Ts:
        Tc = T.copy()
        Tc[:3, 3] = Tc[:3, 3] + shift
        link_Ts_c.append((name, Tc))
    link_clouds_c: List[Tuple[str, np.ndarray]] = []
    for name, pts in link_clouds:
        link_clouds_c.append((name, pts + shift))

    mins, maxs = [], []
    for m in meshes_c:
        aabb = m.get_axis_aligned_bounding_box()
        mins.append(np.asarray(aabb.min_bound))
        maxs.append(np.asarray(aabb.max_bound))
    for _, T in link_Ts_c:
        mins.append(T[:3, 3] - axis_length)
        maxs.append(T[:3, 3] + axis_length)
    for _, pts in link_clouds_c:
        if len(pts):
            mins.append(pts.min(axis=0))
            maxs.append(pts.max(axis=0))
    # Green world axes at new origin (0,0,0)
    mins.append(np.array([-world_axis_length, -world_axis_length, -world_axis_length]))
    maxs.append(np.array([world_axis_length, world_axis_length, world_axis_length]))
    # Ground = world XY plane of green axes (through base-center origin)
    ground_z = 0.0
    mn = np.min(np.stack(mins, axis=0), axis=0)
    mx = np.max(np.stack(maxs, axis=0), axis=0)
    # Expand framing to include ground under/through the origin
    mn = mn.copy()
    mx = mx.copy()
    mn[2] = min(mn[2], ground_z)
    mx[2] = max(mx[2], ground_z)
    center = 0.5 * (mn + mx)
    extent = float(np.linalg.norm(mx - mn))
    cam_radius = max(extent * 1.25, 0.45)
    cube_size = max(extent * cube_scale, 0.0028)
    ground_size = max(float(np.linalg.norm(mx[:2] - mn[:2])) * 1.35, extent * 0.95, 0.80)

    gray_meshes = _to_gray(meshes_c)

    colors = _link_palette(len(link_clouds_c))
    pc_meshes: List[o3d.geometry.TriangleMesh] = []
    for i, (_, pts) in enumerate(link_clouds_c):
        pc_meshes.append(_pc_cubes(pts, colors[i], cube_size))

    # Red: per-link frames
    axis_meshes: List[o3d.geometry.TriangleMesh] = []
    for _, T in link_Ts_c:
        for a in _axis_group(axis_length, axis_radius, COLOR_AXIS):
            aa = o3d.geometry.TriangleMesh(a)
            aa.transform(T)
            axis_meshes.append(aa)

    # Green: world frame at base-center midpoint (= origin after shift)
    world_axes = _axis_group(world_axis_length, world_axis_radius, COLOR_WORLD)

    ground, ground_grid = _make_ground(
        cx=0.0,
        cy=0.0,
        z=float(ground_z),
        size=ground_size,
        n_grid=12,
    )

    renderer = rendering.OffscreenRenderer(width, height)
    scene = renderer.scene
    scene.set_background([1.0, 1.0, 1.0, 1.0])
    # No-shadow lighting profile only (do not re-enable sun shadows afterward)
    sun_dir = np.array([0.35, -0.25, -0.9], dtype=np.float32)
    scene.set_lighting(rendering.Open3DScene.NO_SHADOWS, sun_dir)

    # Ground first (under everything) — unlit so it cannot show cast shadows
    mat_ground = rendering.MaterialRecord()
    mat_ground.shader = "defaultUnlit"
    mat_ground.base_color = [COLOR_GROUND[0], COLOR_GROUND[1], COLOR_GROUND[2], 1.0]
    scene.add_geometry("ground", ground, mat_ground)
    scene.scene.geometry_shadows("ground", False, False)

    mat_grid = rendering.MaterialRecord()
    mat_grid.shader = "unlitLine"
    mat_grid.line_width = 1.0
    mat_grid.base_color = [
        COLOR_GROUND_GRID[0],
        COLOR_GROUND_GRID[1],
        COLOR_GROUND_GRID[2],
        1.0,
    ]
    scene.add_geometry("ground_grid", ground_grid, mat_grid)
    scene.scene.geometry_shadows("ground_grid", False, False)

    mat_mesh = rendering.MaterialRecord()
    mat_mesh.shader = "defaultLitTransparency"
    mat_mesh.base_color = [
        COLOR_MESH[0],
        COLOR_MESH[1],
        COLOR_MESH[2],
        float(np.clip(mesh_alpha, 0.01, 1.0)),
    ]
    mat_mesh.base_roughness = 0.90
    for i, m in enumerate(gray_meshes):
        name = f"part_{i}"
        scene.add_geometry(name, m, mat_mesh)
        scene.scene.geometry_shadows(name, False, False)

    for i, pc in enumerate(pc_meshes):
        if len(np.asarray(pc.vertices)) == 0:
            continue
        c = colors[i]
        mat_i = rendering.MaterialRecord()
        mat_i.shader = "defaultLit"
        mat_i.base_color = [c[0], c[1], c[2], 1.0]
        name = f"pc_{i}"
        scene.add_geometry(name, pc, mat_i)
        scene.scene.geometry_shadows(name, False, False)

    mat_axis = rendering.MaterialRecord()
    mat_axis.shader = "defaultLit"
    mat_axis.base_color = [COLOR_AXIS[0], COLOR_AXIS[1], COLOR_AXIS[2], 1.0]
    for i, a in enumerate(axis_meshes):
        name = f"axis_{i}"
        scene.add_geometry(name, a, mat_axis)
        scene.scene.geometry_shadows(name, False, False)

    mat_world = rendering.MaterialRecord()
    mat_world.shader = "defaultLit"
    mat_world.base_color = [COLOR_WORLD[0], COLOR_WORLD[1], COLOR_WORLD[2], 1.0]
    for i, a in enumerate(world_axes):
        name = f"world_axis_{i}"
        scene.add_geometry(name, a, mat_world)
        scene.scene.geometry_shadows(name, False, False)

    view_list = views[:1] if primary_only else views
    saved: List[Path] = []
    up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    for name, elev, azim in view_list:
        eye = _sphere_eye(center, cam_radius, elev, azim)
        look = center - eye
        look = look / (np.linalg.norm(look) + 1e-9)
        cam_up = up.copy()
        if abs(np.dot(look, up)) > 0.92:
            cam_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        renderer.setup_camera(42.0, center, eye, cam_up)
        img = renderer.render_to_image()
        path = out_dir / f"full_mesh_axes_{name}_e{int(elev)}_a{int(azim)}.png"
        _save_png(path, img)
        saved.append(path)
        print(f"[saved] {path}  (cube_size={cube_size:.4f}, ground_z={ground_z:.3f})")

    del renderer
    return saved

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out_dir",
        type=str,
        default=str(
            Path(__file__).resolve().parents[1] / "docs" / "figures" / "robot_full_mesh_axes"
        ),
    )
    parser.add_argument("--arms", choices=("left", "right", "both"), default="both")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_points", type=int, default=64, help="Surface samples per link")
    parser.add_argument("--axis_length", type=float, default=0.065)
    parser.add_argument("--axis_radius", type=float, default=0.0026)
    parser.add_argument(
        "--world_axis_length",
        type=float,
        default=0.16,
        help="Green world-frame XYZ arrows at base-center midpoint",
    )
    parser.add_argument("--world_axis_radius", type=float, default=0.0035)
    parser.add_argument("--mesh_alpha", type=float, default=MESH_ALPHA)
    parser.add_argument("--edge_alpha", type=float, default=EDGE_ALPHA)
    parser.add_argument(
        "--cube_scale",
        type=float,
        default=0.009,
        help="Cube size as fraction of scene extent (slightly smaller)",
    )
    parser.add_argument("--width", type=int, default=1800)
    parser.add_argument("--height", type=int, default=1400)
    parser.add_argument("--all_views", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    gen = RobotPointCloudGenerator(
        points_per_link=args.n_points,
        sample_region="surface",
        device="cpu",
    )
    qpos = build_demo_qpos(args.seed)
    print("arms:", args.arms)
    print("qpos:", np.round(qpos, 3).tolist())
    print(
        f"n_points={args.n_points}, mesh_alpha={args.mesh_alpha:.3f}, "
        f"link_axis={args.axis_length:.4f}, world_axis={args.world_axis_length:.4f}"
    )

    meshes = assemble_world_meshes(gen, qpos, arms=args.arms)
    link_Ts = _collect_link_Ts(gen, qpos, arms=args.arms)
    link_clouds = _world_link_clouds(gen, qpos, arms=args.arms)
    world_origin = _base_center_midpoint(gen, qpos)
    print(
        f"mesh parts={len(meshes)}, link frames={len(link_Ts)}, "
        f"link clouds={len(link_clouds)}"
    )
    print(
        "world origin = midpoint(fl_base, fr_base) =",
        np.round(world_origin, 4).tolist(),
    )
    if not meshes:
        raise SystemExit("No meshes loaded")

    paths = render_full_arm_with_axes(
        meshes,
        link_Ts,
        link_clouds,
        out_dir,
        DEFAULT_VIEWS,
        axis_length=args.axis_length,
        axis_radius=args.axis_radius,
        world_axis_length=args.world_axis_length,
        world_axis_radius=args.world_axis_radius,
        width=args.width,
        height=args.height,
        primary_only=not args.all_views,
        mesh_alpha=args.mesh_alpha,
        edge_alpha=args.edge_alpha,
        cube_scale=args.cube_scale,
        world_origin=world_origin,
    )

    primary = paths[0]
    canon = out_dir / "full_arm_mesh_with_link_axes.png"
    if primary.resolve() != canon.resolve():
        canon.write_bytes(primary.read_bytes())
        print(f"[saved] {canon}")

    # Color legend
    colors = _link_palette(len(link_clouds))
    legend_lines = ["| # | Link | Color (RGB) |", "|---|------|-------------|"]
    for i, (name, _) in enumerate(link_clouds):
        c = colors[i]
        legend_lines.append(
            f"| {i} | `{name}` | ({c[0]:.3f}, {c[1]:.3f}, {c[2]:.3f}) |"
        )

    readme = out_dir / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# Full-arm ghost mesh + per-link colored PC",
                "",
                "Almost-transparent gray mesh (**no edges**) · **colored cubes** per link · "
                "**red** link XYZ · **green** world XYZ @ **base-center midpoint** · "
                "**ground on world XY plane** (green X/Y).",
                "",
                f"- arms=`{args.arms}`, seed={args.seed}, n_points=**{args.n_points}**",
                f"- mesh_alpha=**{args.mesh_alpha:.3f}**, cube_scale=**{args.cube_scale:.3f}**",
                f"- link axis_length=**{args.axis_length:.3f}** m",
                f"- world axis_length=**{args.world_axis_length:.3f}** m (green)",
                f"- world origin (pre-shift) = midpoint(fl_base, fr_base) = "
                f"`{np.round(world_origin, 4).tolist()}`",
                "- ground_z (after shift) = **0.000** (world XY plane)",
                "",
                f"Main figure: `{canon.name}`",
                "",
                "## Link colors",
                "",
                *legend_lines,
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"Done → {out_dir}")

if __name__ == "__main__":
    main()
