#!/usr/bin/env python3
"""Export full-arm visual mesh renders (robot native colors, white bg, no axes).

Multiple camera viewpoints so you can pick one for the paper.

Usage:
  python policy/SPACE/scripts/export_robot_mesh_views.py \\
      --out_dir policy/SPACE/docs/figures/robot_mesh_views \\
      --arms both
"""

from __future__ import annotations

import argparse
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import trimesh

# Headless EGL before importing Open3D rendering
os.environ.setdefault("EGL_PLATFORM", "surfaceless")

import open3d as o3d
from open3d.visualization import rendering

BLOCK_DIR = Path(__file__).resolve().parents[1] / "block"
if str(BLOCK_DIR) not in sys.path:
    sys.path.insert(0, str(BLOCK_DIR))

from make_robot import (  # noqa: E402
    RobotPointCloudGenerator,
    _np_origin_matrix,
    _parse_floats,
)

# (name, elev_deg, azim_deg) — Open3D camera: look from spherical direction
DEFAULT_VIEWS: List[Tuple[str, float, float]] = [
    ("front", 15.0, 0.0),
    ("front_left", 18.0, 35.0),
    ("front_right", 18.0, -35.0),
    ("side_left", 12.0, 90.0),
    ("side_right", 12.0, -90.0),
    ("top", 75.0, 25.0),
    ("three_quarter_L", 22.0, 45.0),
    ("three_quarter_R", 22.0, -45.0),
    ("back", 15.0, 180.0),
    ("low_front", 5.0, 20.0),
]

def build_demo_qpos(seed: int = 0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    q = rng.uniform(-0.35, 0.35, size=14).astype(np.float32)
    q[6] = 0.55
    q[13] = 0.55
    q[2] = 0.55
    q[3] = -0.40
    q[4] = 0.35
    # Mirror-ish right arm for a symmetric dual-arm pose
    q[9] = 0.55
    q[10] = -0.40
    q[11] = 0.35
    return q

def _rgba01_from_material(mat) -> np.ndarray:
    """Extract RGB(A) in [0,1] from trimesh PBR / simple material."""
    if mat is None:
        return np.array([0.35, 0.35, 0.35, 1.0], dtype=np.float64)

    tex = getattr(mat, "baseColorTexture", None)
    if tex is not None:
        img = getattr(tex, "image", None) or tex
        try:
            arr = np.asarray(img)
            if arr.ndim >= 2 and arr.size > 0:
                flat = arr.reshape(-1, arr.shape[-1]).astype(np.float64)
                mean = flat[:, :3].mean(axis=0)
                if mean.max() > 1.0:
                    mean = mean / 255.0
                # Tiny near-black textures (AgileX DAE) → dark body color
                alpha = 1.0
                if flat.shape[1] > 3:
                    a = flat[:, 3].mean()
                    alpha = float(a / 255.0 if a > 1.0 else a)
                return np.array([*mean.tolist(), alpha], dtype=np.float64)
        except Exception:
            pass

    bcf = getattr(mat, "baseColorFactor", None)
    if bcf is not None:
        c = np.asarray(bcf, dtype=np.float64).reshape(-1)
        if c.max() > 1.0:
            c = c / 255.0
        if c.size == 3:
            c = np.array([c[0], c[1], c[2], 1.0], dtype=np.float64)
        return c[:4]

    mc = getattr(mat, "main_color", None)
    if mc is not None:
        c = np.asarray(mc, dtype=np.float64).reshape(-1)
        if c.max() > 1.0:
            c = c / 255.0
        if c.size == 3:
            c = np.array([c[0], c[1], c[2], 1.0], dtype=np.float64)
        return c[:4]

    return np.array([0.35, 0.35, 0.35, 1.0], dtype=np.float64)

def _paint_mesh_from_visual(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Bake material / texture mean into face colors (for Open3D)."""
    mesh = mesh.copy()
    rgba = np.array([0.35, 0.35, 0.35, 1.0], dtype=np.float64)
    vis = getattr(mesh, "visual", None)
    if vis is not None:
        kind = getattr(vis, "kind", None)
        if kind == "texture":
            rgba = _rgba01_from_material(getattr(vis, "material", None))
        elif hasattr(vis, "face_colors") and len(getattr(vis, "face_colors", [])) > 0:
            fc = np.asarray(vis.face_colors, dtype=np.float64)
            if fc.max() > 1.0:
                fc = fc / 255.0
            # Use mean if varied, else first
            rgba = fc.mean(axis=0)[:4] if fc.ndim == 2 else fc[:4]
        else:
            rgba = _rgba01_from_material(getattr(vis, "material", None))

    rgba_u8 = (np.clip(rgba, 0, 1) * 255).astype(np.uint8)
    mesh.visual = trimesh.visual.ColorVisuals(
        mesh=mesh, face_colors=np.tile(rgba_u8, (len(mesh.faces), 1))
    )
    return mesh

def _load_visual_parts(path: Path) -> List[trimesh.Trimesh]:
    """Load DAE/STL visual file → list of colored trimeshes in file-local frame."""
    if not path.exists():
        return []
    loaded = trimesh.load(str(path), process=False, force=None)
    parts: List[trimesh.Trimesh] = []
    if isinstance(loaded, trimesh.Scene):
        # dump() applies scene-graph transforms
        dumped = loaded.dump(concatenate=False)
        if not isinstance(dumped, list):
            dumped = [dumped]
        for g in dumped:
            if isinstance(g, trimesh.Trimesh) and len(g.faces) > 0:
                parts.append(_paint_mesh_from_visual(g))
    elif isinstance(loaded, trimesh.Trimesh) and len(loaded.faces) > 0:
        parts.append(_paint_mesh_from_visual(loaded))
    return parts

def _visual_origin(link_el: ET.Element) -> np.ndarray:
    vis = link_el.find("visual")
    if vis is None:
        return np.eye(4, dtype=np.float64)
    origin_el = vis.find("origin")
    if origin_el is None:
        return np.eye(4, dtype=np.float64)
    xyz = np.array(_parse_floats(origin_el.get("xyz", "0 0 0")), dtype=np.float64)
    rpy = np.array(_parse_floats(origin_el.get("rpy", "0 0 0")), dtype=np.float64)
    return _np_origin_matrix(xyz, rpy)

def load_arm_visual_meshes(
    gen: RobotPointCloudGenerator, link_names: Sequence[str]
) -> Dict[str, List[trimesh.Trimesh]]:
    root = ET.parse(gen.urdf_path).getroot()
    urdf_dir = gen.urdf_path.parent
    out: Dict[str, List[trimesh.Trimesh]] = {}
    for name in link_names:
        link_el = root.find(f"./link[@name='{name}']")
        if link_el is None:
            continue
        local_tf = _visual_origin(link_el)
        parts: List[trimesh.Trimesh] = []
        for vis in link_el.findall("visual"):
            mesh_el = vis.find("geometry/mesh")
            if mesh_el is None:
                continue
            fn = mesh_el.get("filename")
            if not fn:
                continue
            for part in _load_visual_parts((urdf_dir / fn).resolve()):
                m = part.copy()
                m.apply_transform(local_tf)
                parts.append(m)
        if parts:
            out[name] = parts
    return out

def _trimesh_to_o3d(mesh: trimesh.Trimesh) -> o3d.geometry.TriangleMesh:
    tm = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.asarray(mesh.vertices, dtype=np.float64)),
        o3d.utility.Vector3iVector(np.asarray(mesh.faces, dtype=np.int32)),
    )
    fc = np.asarray(mesh.visual.face_colors, dtype=np.float64)
    if fc.max() > 1.0:
        fc = fc / 255.0
    # face color → vertex color (duplicate ok for flat look)
    vc = np.zeros((len(mesh.vertices), 3), dtype=np.float64)
    counts = np.zeros(len(mesh.vertices), dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    for i, f in enumerate(faces):
        c = fc[i, :3]
        for vi in f:
            vc[vi] += c
            counts[vi] += 1.0
    counts = np.maximum(counts, 1.0)
    vc = vc / counts[:, None]
    tm.vertex_colors = o3d.utility.Vector3dVector(vc)
    tm.compute_vertex_normals()
    return tm

def assemble_world_meshes(
    gen: RobotPointCloudGenerator,
    qpos: np.ndarray,
    arms: str = "both",
) -> List[o3d.geometry.TriangleMesh]:
    fk = gen.forward_kinematics(torch.as_tensor(qpos, dtype=torch.float32), arm=arms)
    o3d_meshes: List[o3d.geometry.TriangleMesh] = []

    arm_items: List[Tuple[str, Sequence[str]]] = []
    if arms in ("left", "both"):
        arm_items.append(("left", gen.left_links))
    if arms in ("right", "both"):
        arm_items.append(("right", gen.right_links))

    for arm_key, links in arm_items:
        visual_map = load_arm_visual_meshes(gen, links)
        for name in links:
            if name not in visual_map or name not in fk[arm_key]:
                continue
            T = fk[arm_key][name][0].detach().cpu().numpy().astype(np.float64)
            for part in visual_map[name]:
                m = part.copy()
                m.apply_transform(T)
                o3d_meshes.append(_trimesh_to_o3d(m))
    return o3d_meshes

def _sphere_eye(center: np.ndarray, radius: float, elev_deg: float, azim_deg: float) -> np.ndarray:
    elev = np.deg2rad(elev_deg)
    azim = np.deg2rad(azim_deg)
    # azim=0 → +X (front-ish after embodiment yaw); elev=0 → horizon
    x = radius * np.cos(elev) * np.cos(azim)
    y = radius * np.cos(elev) * np.sin(azim)
    z = radius * np.sin(elev)
    return center + np.array([x, y, z], dtype=np.float64)

def render_views(
    meshes: List[o3d.geometry.TriangleMesh],
    out_dir: Path,
    views: Sequence[Tuple[str, float, float]],
    width: int = 1600,
    height: int = 1200,
    dpi_note: int = 300,
) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Combined AABB for framing
    mins = []
    maxs = []
    for m in meshes:
        aabb = m.get_axis_aligned_bounding_box()
        mins.append(np.asarray(aabb.min_bound))
        maxs.append(np.asarray(aabb.max_bound))
    mn = np.min(np.stack(mins, axis=0), axis=0)
    mx = np.max(np.stack(maxs, axis=0), axis=0)
    center = 0.5 * (mn + mx)
    extent = float(np.linalg.norm(mx - mn))
    radius = max(extent * 1.15, 0.35)

    renderer = rendering.OffscreenRenderer(width, height)
    scene = renderer.scene
    scene.set_background([1.0, 1.0, 1.0, 1.0])
    scene.scene.set_sun_light([0.4, -0.3, -0.9], [1.0, 1.0, 1.0], 80000)
    scene.scene.enable_sun_light(True)

    mat = rendering.MaterialRecord()
    mat.shader = "defaultLit"
    for i, m in enumerate(meshes):
        scene.add_geometry(f"part_{i}", m, mat)

    saved: List[Path] = []
    up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    fov = 45.0
    for name, elev, azim in views:
        eye = _sphere_eye(center, radius, elev, azim)
        # Avoid degenerate up when looking nearly along +Z
        look = center - eye
        look = look / (np.linalg.norm(look) + 1e-9)
        cam_up = up.copy()
        if abs(np.dot(look, up)) > 0.92:
            cam_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        renderer.setup_camera(fov, center, eye, cam_up)
        img = renderer.render_to_image()
        path = out_dir / f"mesh_{name}_e{int(elev)}_a{int(azim)}.png"
        o3d.io.write_image(str(path), img)
        saved.append(path)
        print(f"[saved] {path}")

    # Also write a contact-sheet style overview (matplotlib)
    try:
        import matplotlib.pyplot as plt
        from PIL import Image

        n = len(saved)
        cols = min(5, n)
        rows = int(np.ceil(n / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 2.6 * rows))
        axes = np.atleast_1d(axes).ravel()
        for ax in axes:
            ax.axis("off")
        for ax, p in zip(axes, saved):
            ax.imshow(Image.open(p))
            ax.set_title(p.stem.replace("mesh_", ""), fontsize=8)
            ax.axis("off")
        fig.patch.set_facecolor("white")
        sheet = out_dir / "00_all_views_contact_sheet.png"
        fig.tight_layout(pad=0.3)
        fig.savefig(sheet, dpi=dpi_note, facecolor="white", bbox_inches="tight")
        plt.close(fig)
        print(f"[saved] {sheet}")
        saved.insert(0, sheet)
    except Exception as e:
        print(f"[warn] contact sheet skipped: {e}")

    del renderer
    return saved

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out_dir",
        type=str,
        default=str(
            Path(__file__).resolve().parents[1] / "docs" / "figures" / "robot_mesh_views"
        ),
    )
    parser.add_argument("--arms", choices=("left", "right", "both"), default="both")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=1200)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    gen = RobotPointCloudGenerator(points_per_link=4, sample_region="surface", device="cpu")
    qpos = build_demo_qpos(args.seed)
    print("arms:", args.arms)
    print("qpos:", np.round(qpos, 3).tolist())

    meshes = assemble_world_meshes(gen, qpos, arms=args.arms)
    print(f"loaded {len(meshes)} visual mesh parts")
    if not meshes:
        raise SystemExit("No visual meshes loaded")

    paths = render_views(meshes, out_dir, DEFAULT_VIEWS, width=args.width, height=args.height)

    readme = out_dir / "README.md"
    lines = [
        "# Full-arm visual mesh views",
        "",
        "Robot **native** DAE/PBR colors · white background · no axes.",
        "",
        f"Generated by `export_robot_mesh_views.py` (arms=`{args.arms}`, seed={args.seed}).",
        "",
        "| File | View |",
        "|------|------|",
    ]
    for name, elev, azim in DEFAULT_VIEWS:
        lines.append(f"| `mesh_{name}_e{int(elev)}_a{int(azim)}.png` | elev={elev}, azim={azim} |")
    lines.append("")
    lines.append("Pick one for the paper; see also `00_all_views_contact_sheet.png`.")
    readme.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Done → {out_dir} ({len(paths)} files)")

if __name__ == "__main__":
    main()
