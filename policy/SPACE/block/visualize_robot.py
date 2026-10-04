#!/usr/bin/env python3
"""Export lightweight HTML with interactive 3D point clouds (Three.js, orbit-drag).

No PNG. Topology: one HTML, each relation in its own panel with labels.

Usage:
  python policy/SPACE/block/visualize_robot.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import torch

BLOCK_DIR = Path(__file__).resolve().parent
if str(BLOCK_DIR) not in sys.path:
    sys.path.insert(0, str(BLOCK_DIR))

from make_robot import RobotPointCloudGenerator, _transform_local_points  # noqa: E402

# Distinct colors per link index (base→link8); shared by L/R legends.
_LINK_PALETTE = [
    "#ef4444",  # 0 base
    "#f97316",  # 1
    "#eab308",  # 2
    "#84cc16",  # 3
    "#22c55e",  # 4
    "#14b8a6",  # 5
    "#3b82f6",  # 6 wrist
    "#a855f7",  # 7 finger
    "#ec4899",  # 8 finger
]

def _to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()

def _pts_json(pts: np.ndarray, decimals: int = 5) -> str:
    arr = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
    return json.dumps(np.round(arr, decimals).tolist(), separators=(",", ":"))

# Three.js r160 + OrbitControls via importmap (no build step).
_CSS = """
:root { color-scheme: dark; }
body { margin:0; font-family: ui-sans-serif, system-ui, sans-serif; background:#0b0d10; color:#e6ebf2; }
h1 { font-size:1.25rem; margin:1rem 1.25rem 0.25rem; }
h2 { font-size:1.0rem; margin:0 0 0.4rem; line-height:1.35; }
.meta { color:#9aa3b2; margin:0 1.25rem 1rem; font-size:0.9rem; }
.panel { border:1px solid #2a2f3a; border-radius:10px; margin:1rem 1.25rem; padding:0.85rem 1rem 1rem; background:#12151b; }
.badge { display:inline-block; padding:0.15rem 0.5rem; border-radius:999px; font-size:0.75rem; margin-right:0.4rem; }
.badge.L { background:#1b3a4a; color:#7dd3fc; }
.badge.R { background:#3a1b2e; color:#f9a8d4; }
.rel { color:#fbbf24; font-weight:600; }
.viewer { width:100%; height:360px; border:1px solid #2a2f3a; border-radius:8px; background:#0a0c10; touch-action:none; }
.viewer.tall { height:480px; }
.legend { display:flex; flex-wrap:wrap; gap:0.45rem 0.85rem; margin:0.5rem 0 0.75rem; font-size:0.8rem; color:#cbd5e1; }
.legend span { display:inline-flex; align-items:center; gap:0.35rem; }
.swatch { width:12px; height:12px; border-radius:3px; display:inline-block; }
.hint { color:#6b7280; font-size:0.8rem; margin-top:0.35rem; }
.nav a { color:#7dd3fc; margin-right:1rem; }
.toc { margin:0.5rem 1.25rem 1rem; columns:2; max-width:960px; }
.toc a { color:#9aa3b2; font-size:0.85rem; display:block; text-decoration:none; }
.toc a:hover { color:#e6ebf2; }
"""

_VIEWER_MODULE = r"""
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// Browsers cap concurrent WebGL contexts (~8–16). Topology pages have 36 panels.
// Keep a small LRU of live contexts; create on enter-viewport, dispose on leave.
const MAX_LIVE = 6;
const live = []; // [{container, dispose}]

function fitCamera(camera, controls, positions) {
  if (!positions.length) return;
  const box = new THREE.Box3();
  const v = new THREE.Vector3();
  for (let i = 0; i < positions.length; i += 3) {
    v.set(positions[i], positions[i+1], positions[i+2]);
    box.expandByPoint(v);
  }
  const size = new THREE.Vector3();
  const center = new THREE.Vector3();
  box.getSize(size);
  box.getCenter(center);
  const radius = Math.max(size.length() * 0.5, 0.05);
  const dist = radius * 2.8 / Math.tan(THREE.MathUtils.degToRad(camera.fov * 0.5));
  camera.position.set(center.x + dist * 0.75, center.y + dist * 0.55, center.z + dist * 0.75);
  camera.near = Math.max(dist / 500, 1e-4);
  camera.far = Math.max(dist * 50, 50);
  camera.updateProjectionMatrix();
  controls.target.copy(center);
  controls.minDistance = radius * 0.05;
  controls.maxDistance = dist * 20;
  controls.update();
}

function showPlaceholder(container, msg) {
  container.innerHTML = '';
  const d = document.createElement('div');
  d.style.cssText = 'display:flex;align-items:center;justify-content:center;height:100%;color:#6b7280;font-size:0.85rem;';
  d.textContent = msg || 'scroll into view to render';
  container.appendChild(d);
}

function createLive(container, pts, colorHex, colorList) {
  const width = container.clientWidth || 640;
  const height = container.clientHeight || 360;
  let renderer;
  try {
    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
  } catch (err) {
    showPlaceholder(container, 'WebGL unavailable');
    console.warn(err);
    return null;
  }
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setSize(width, height);
  renderer.setClearColor(0x0a0c10, 1);
  container.innerHTML = '';
  container.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(50, width / height, 0.001, 1000);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.rotateSpeed = 0.9;
  controls.zoomSpeed = 1.1;

  const n = pts.length;
  const positions = new Float32Array(n * 3);
  const colors = new Float32Array(n * 3);
  const c = new THREE.Color(colorHex || 0x5ec8ff);
  let minx=Infinity,maxx=-Infinity,miny=Infinity,maxy=-Infinity,minz=Infinity,maxz=-Infinity;
  for (let i = 0; i < n; i++) {
    const x = pts[i][0], y = pts[i][1], z = pts[i][2];
    positions[i*3] = x; positions[i*3+1] = y; positions[i*3+2] = z;
    if(x<minx)minx=x; if(x>maxx)maxx=x;
    if(y<miny)miny=y; if(y>maxy)maxy=y;
    if(z<minz)minz=z; if(z>maxz)maxz=z;
    if (colorList && colorList[i]) {
      const cc = new THREE.Color(colorList[i]);
      colors[i*3] = cc.r; colors[i*3+1] = cc.g; colors[i*3+2] = cc.b;
    } else {
      colors[i*3] = c.r; colors[i*3+1] = c.g; colors[i*3+2] = c.b;
    }
  }

  const geom = new THREE.BufferGeometry();
  geom.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geom.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  const mat = new THREE.PointsMaterial({
    size: 5,
    sizeAttenuation: false,
    vertexColors: true,
  });
  scene.add(new THREE.Points(geom, mat));

  let extent = 0.05;
  if (n > 0) extent = Math.max(maxx-minx, maxy-miny, maxz-minz, 0.05);
  scene.add(new THREE.AxesHelper(extent * 0.4));
  scene.add(new THREE.GridHelper(extent * 2.5, 10, 0x334155, 0x1f2937));

  if (n === 0) console.warn('empty point cloud', container.id);
  fitCamera(camera, controls, positions);

  let frame = 0;
  function animate() {
    frame = requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
  }
  animate();

  const onResize = () => {
    const w = container.clientWidth || 640;
    const h = container.clientHeight || 360;
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h);
  };
  window.addEventListener('resize', onResize);

  function dispose() {
    cancelAnimationFrame(frame);
    frame = 0;
    window.removeEventListener('resize', onResize);
    controls.dispose();
    geom.dispose();
    mat.dispose();
    renderer.dispose();
    if (renderer.forceContextLoss) renderer.forceContextLoss();
    showPlaceholder(container, 'scroll into view to render');
  }

  return { container, dispose };
}

function acquire(container, pts, colorHex, colorList) {
  // already live?
  const idx = live.findIndex((x) => x.container === container);
  if (idx >= 0) {
    const cur = live.splice(idx, 1)[0];
    live.push(cur); // refresh LRU
    return;
  }
  while (live.length >= MAX_LIVE) {
    const old = live.shift();
    old.dispose();
  }
  const created = createLive(container, pts, colorHex, colorList);
  if (created) live.push(created);
}

export function mountPointCloud(container, pts, colorHex, colorList) {
  if (!container) return;
  showPlaceholder(container, 'scroll into view to render');
  const io = new IntersectionObserver((entries) => {
    entries.forEach((e) => {
      if (e.isIntersecting) {
        acquire(container, pts, colorHex, colorList);
      } else {
        const idx = live.findIndex((x) => x.container === container);
        if (idx >= 0) {
          const cur = live.splice(idx, 1)[0];
          cur.dispose();
        }
      }
    });
  }, { root: null, rootMargin: '120px 0px', threshold: 0.01 });
  io.observe(container);
}

window.mountPointCloud = mountPointCloud;
"""

def _html_shell(title: str, body: str, boot_script: str) -> str:
    """boot_script runs after module loads; may call mountPointCloud(...)."""
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>{title}</title>
<style>{_CSS}</style>
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
{body}
<script type="module">
{_VIEWER_MODULE}
{boot_script}
</script>
</body>
</html>
"""

def _try_gt_simulator_pc(qpos: np.ndarray, gen: RobotPointCloudGenerator) -> np.ndarray:
    try:
        import sapien.core as sapien

        engine = sapien.Engine()
        renderer = sapien.SapienRenderer()
        engine.set_renderer(renderer)
        scene = engine.create_scene()
        scene.set_timestep(1 / 250)
        loader = scene.create_urdf_loader()
        loader.fix_root_link = True
        robot = loader.load(str(gen.urdf_path))

        # Must match make_robot world_root / RoboTwin embodiment robot_pose.
        # pose7 = [x, y, z, qw, qx, qy, qz]  →  sapien.Pose(p, q=[w,x,y,z])
        pose7 = gen.robot_config.get("robot_pose", [[0, -0.65, 0.0, 0.707, 0, 0, 0.707]])[0]
        robot.set_pose(sapien.Pose(pose7[:3], pose7[3:]))

        qfull = robot.get_qpos()
        names = robot.get_active_joints()
        name_to_idx = {j.get_name(): i for i, j in enumerate(names)}
        mapping = []
        for i, jn in enumerate(
            [f"fl_joint{k}" for k in range(1, 7)]
            + ["fl_joint7"]
            + [f"fr_joint{k}" for k in range(1, 7)]
            + ["fr_joint7"]
        ):
            if jn in name_to_idx:
                mapping.append((name_to_idx[jn], float(qpos[i])))
        q = np.array(qfull, dtype=np.float32)
        for idx, val in mapping:
            if idx < len(q):
                q[idx] = val
        robot.set_qpos(q)
        scene.step()
        pose_map = {link.get_name(): link.get_pose() for link in robot.get_links()}

        def pose_to_mat(pose) -> np.ndarray:
            mat = np.eye(4, dtype=np.float64)
            mat[:3, 3] = pose.p
            w, x, y, z = pose.q
            xx, yy, zz = x * x, y * y, z * z
            xy, xz, yz = x * y, x * z, y * z
            wx, wy, wz = w * x, w * y, w * z
            mat[:3, :3] = np.array(
                [
                    [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
                    [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
                    [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
                ]
            )
            return mat

        pts = []
        for arm_links, locals_list in (
            (gen.left_links, gen.left_local),
            (gen.right_links, gen.right_local),
        ):
            for name, pts_local in zip(arm_links, locals_list):
                if name not in pose_map:
                    continue
                T = pose_to_mat(pose_map[name])
                pl = pts_local.detach().cpu().numpy()
                homo = np.concatenate([pl, np.ones((pl.shape[0], 1))], axis=1)
                pts.append((T @ homo.T).T[:, :3])
        if pts:
            gt = np.concatenate(pts, axis=0).astype(np.float32)
            geo = gen.generate_geometry_pc(torch.from_numpy(qpos).float())
            geo_np = np.concatenate(
                [_to_numpy(geo["left_geometry_pc"]), _to_numpy(geo["right_geometry_pc"])],
                axis=0,
            )
            n = min(len(gt), len(geo_np))
            err = float(np.linalg.norm(gt[:n] - geo_np[:n], axis=1).mean())
            print(f"[visualize] GT with set_pose · mean point L2 vs FK ≈ {err:.6f} m")
            return gt
    except Exception as e:
        print(f"[visualize] Sapien GT unavailable ({e}); FK geometry as GT proxy.")

    out = gen.generate_geometry_pc(torch.from_numpy(qpos).float())
    return np.concatenate(
        [_to_numpy(out["left_geometry_pc"]), _to_numpy(out["right_geometry_pc"])],
        axis=0,
    )

def write_gt_html(path: Path, gt: np.ndarray) -> None:
    body = f"""
<a class="nav" href="index.html">← index</a>
<h1>01 · GT robot point cloud (3D)</h1>
<p class="meta">N={len(gt)} · drag to orbit · scroll to zoom · right-drag to pan</p>
<div class="panel">
  <div id="v0" class="viewer"></div>
  <p class="hint">Axes: X red · Y green · Z blue</p>
</div>
"""
    boot = f"mountPointCloud(document.getElementById('v0'), {_pts_json(gt)}, 0x5ec8ff, null);"
    path.write_text(_html_shell("GT 3D", body, boot), encoding="utf-8")

def write_geometry_html(path: Path, left: np.ndarray, right: np.ndarray) -> None:
    both = np.concatenate([left, right], axis=0)
    color_list = ["#38bdf8"] * len(left) + ["#f472b6"] * len(right)
    body = f"""
<a class="nav" href="index.html">← index</a>
<h1>02 · Geometry PC (3D)</h1>
<p class="meta">
  <span class="badge L">LEFT</span> N={len(left)} &nbsp;
  <span class="badge R">RIGHT</span> N={len(right)} &nbsp;
  world xyz · drag to rotate
</p>
<div class="panel">
  <h2>Left + Right (cyan / pink)</h2>
  <div id="both" class="viewer"></div>
</div>
<div class="panel">
  <h2><span class="badge L">LEFT</span> left_geometry_pc</h2>
  <div id="L" class="viewer"></div>
</div>
<div class="panel">
  <h2><span class="badge R">RIGHT</span> right_geometry_pc</h2>
  <div id="R" class="viewer"></div>
</div>
"""
    boot = f"""
mountPointCloud(document.getElementById('both'), {_pts_json(both)}, 0xffffff, {json.dumps(color_list)});
mountPointCloud(document.getElementById('L'), {_pts_json(left)}, 0x38bdf8, null);
mountPointCloud(document.getElementById('R'), {_pts_json(right)}, 0xf472b6, null);
"""
    path.write_text(_html_shell("Geometry 3D", body, boot), encoding="utf-8")

def _per_link_world_clouds(
    gen: RobotPointCloudGenerator, qpos: np.ndarray
) -> tuple[List[np.ndarray], List[np.ndarray]]:
    """World-frame surface samples for each topology link (same FK as geometry)."""
    q = torch.as_tensor(qpos, dtype=torch.float32).unsqueeze(0)
    fk = gen.forward_kinematics(q)
    left_clouds: List[np.ndarray] = []
    right_clouds: List[np.ndarray] = []
    for clouds, links, locals_list, Tmap in (
        (left_clouds, gen.left_links, gen.left_local, fk["left"]),
        (right_clouds, gen.right_links, gen.right_local, fk["right"]),
    ):
        for name, pts_local in zip(links, locals_list):
            pw = _transform_local_points(Tmap[name], pts_local)[0]
            clouds.append(_to_numpy(pw))
    return left_clouds, right_clouds

def _legend_html(links: Sequence[str]) -> str:
    items = []
    for i, name in enumerate(links):
        color = _LINK_PALETTE[i % len(_LINK_PALETTE)]
        short = name.split("_", 1)[-1] if "_" in name else name
        items.append(
            f'<span><i class="swatch" style="background:{color}"></i>'
            f"[{i}] {short}</span>"
        )
    return '<div class="legend">' + "".join(items) + "</div>"

def _merge_colored_clouds(clouds: Sequence[np.ndarray]) -> tuple[np.ndarray, List[str]]:
    pts_list = []
    colors: List[str] = []
    for i, c in enumerate(clouds):
        color = _LINK_PALETTE[i % len(_LINK_PALETTE)]
        pts_list.append(c)
        colors.extend([color] * len(c))
    if not pts_list:
        return np.zeros((0, 3), dtype=np.float32), []
    return np.concatenate(pts_list, axis=0), colors

def write_topology_html(
    path: Path,
    left_topo: np.ndarray,
    right_topo: np.ndarray,
    left_links: Sequence[str],
    right_links: Sequence[str],
    left_neighbor_pairs: Sequence[tuple],
    right_neighbor_pairs: Sequence[tuple],
    left_world_by_link: Sequence[np.ndarray],
    right_world_by_link: Sequence[np.ndarray],
) -> None:
    sections: List[str] = []
    toc: List[str] = []
    boots: List[str] = []
    pid = 0

    # ---- World overview: all topology links, color = link id ----
    left_all, left_colors = _merge_colored_clouds(left_world_by_link)
    right_all, right_colors = _merge_colored_clouds(right_world_by_link)
    both_all = np.concatenate([left_all, right_all], axis=0) if len(left_all) and len(right_all) else left_all
    # Right arm: same palette but slightly darker via same colors (legend separates arms)
    both_colors = left_colors + right_colors

    toc.append('<a href="#world-overview">WORLD · per-link overview</a>')
    sections.append('<h1 id="world-overview">WORLD · topology links (FK geometry)</h1>')
    sections.append(
        '<p class="meta">Same local samples as topology, transformed to <b>world</b> frame '
        "(identical FK as geometry). Color = link index. Use this to verify arm chain / "
        "gripper 6–7–8 layout before inspecting relative panels.</p>"
    )
    sections.append(
        f'<div class="panel" id="ov-both">'
        f"<h2>Left + Right · all links</h2>"
        f"{_legend_html(left_links)}"
        f'<p class="meta">Left &amp; Right share the same color→index map '
        f"(base=red … link8=pink).</p>"
        f'<div id="ov_both_v" class="viewer tall"></div>'
        f"</div>"
    )
    boots.append(
        f"mountPointCloud(document.getElementById('ov_both_v'), {_pts_json(both_all)}, "
        f"0xffffff, {json.dumps(both_colors)});"
    )
    sections.append(
        f'<div class="panel" id="ov-L">'
        f'<h2><span class="badge L">LEFT</span> per-link world PC</h2>'
        f"{_legend_html(left_links)}"
        f'<div id="ov_L_v" class="viewer tall"></div>'
        f"</div>"
    )
    boots.append(
        f"mountPointCloud(document.getElementById('ov_L_v'), {_pts_json(left_all)}, "
        f"0xffffff, {json.dumps(left_colors)});"
    )
    sections.append(
        f'<div class="panel" id="ov-R">'
        f'<h2><span class="badge R">RIGHT</span> per-link world PC</h2>'
        f"{_legend_html(right_links)}"
        f'<div id="ov_R_v" class="viewer tall"></div>'
        f"</div>"
    )
    boots.append(
        f"mountPointCloud(document.getElementById('ov_R_v'), {_pts_json(right_all)}, "
        f"0xffffff, {json.dumps(right_colors)});"
    )

    for arm_tag, arm_label, badge, topo, links, pairs, color in (
        ("L", "LEFT", "L", left_topo, left_links, left_neighbor_pairs, "0x38bdf8"),
        ("R", "RIGHT", "R", right_topo, right_links, right_neighbor_pairs, "0xf472b6"),
    ):
        assert topo.shape[0] == len(pairs), (topo.shape[0], len(pairs))
        sections.append(f'<h1 id="arm-{arm_tag}">{arm_label} arm topology (relative)</h1>')
        sections.append(
            '<p class="meta">Each panel = one directed relation (not overlaid). '
            "xyz is in <b>neighbor</b> frame. Terminal gripper edges: "
            "<b>6↔7, 6↔8, 7↔8</b>. Drag to orbit.</p>"
        )
        for k, (self_id, nei_id) in enumerate(pairs):
            self_name = links[self_id]
            nei_name = links[nei_id]
            pts = topo[k, :, :3]
            extent = float(np.ptp(pts, axis=0).max()) if pts.size else 0.0
            aid = f"p{pid}"
            pid += 1
            title = (
                f'<span class="badge {badge}">{arm_label}</span> '
                f'rel[{k}] · <span class="rel">{self_name}</span> '
                f'relative to <span class="rel">{nei_name}</span>'
                f' <span class="meta">(self={self_id} → neighbor={nei_id})</span>'
            )
            toc.append(
                f'<a href="#{aid}">{arm_label} [{k}] {self_name} → {nei_name}</a>'
            )
            sections.append(
                f'<div class="panel" id="{aid}">'
                f"<h2>{title}</h2>"
                f'<p class="meta">Formula: <code>p_world = T_i @ p_local</code> (same world FK as geometry), '
                f'then <code>p_j = inv(T_j) @ p_world</code>. '
                f'N={pts.shape[0]} · bbox_extent≈{extent:.4f} m</p>'
                f'<div id="{aid}_v" class="viewer"></div>'
                f'<p class="hint">drag=rotate · scroll=zoom · right-drag=pan</p>'
                f"</div>"
            )
            boots.append(
                f"mountPointCloud(document.getElementById('{aid}_v'), {_pts_json(pts)}, {color}, null);"
            )

    body = f"""
<a class="nav" href="index.html">← index</a>
<h1>03 · Topology PC (3D, ordered)</h1>
<p class="meta">Scroll / TOC · overview first (world, colored by link) · then relative panels · WebGL lazy-loads (max 6 live)</p>
<div class="toc">{"".join(toc)}</div>
{"".join(sections)}
"""
    path.write_text(_html_shell("Topology 3D", body, "\n".join(boots)), encoding="utf-8")

def write_index(path: Path) -> None:
    path.write_text(
        _html_shell(
            "Robot PC 3D index",
            """
<h1>Interactive 3D robot point clouds</h1>
<p class="meta">Three.js Points + OrbitControls · needs network for CDN on first load</p>
<div class="panel">
  <p><a href="01_gt.html">01_gt.html</a> — GT</p>
  <p><a href="02_geometry.html">02_geometry.html</a> — geometry L/R</p>
  <p><a href="03_topology.html">03_topology.html</a> — world per-link overview + topology relations</p>
</div>
""",
            "",
        ),
        encoding="utf-8",
    )

def visualize(
    qpos: Optional[np.ndarray] = None,
    out_dir: str | Path = "viz_out",
    points_per_link: int = 48,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gen = RobotPointCloudGenerator(points_per_link=points_per_link, device="cpu")
    if qpos is None:
        rng = np.random.RandomState(0)
        qpos = rng.uniform(-0.4, 0.4, size=14).astype(np.float32)
        qpos[6] = qpos[13] = 0.4
    qpos = np.asarray(qpos, dtype=np.float32)

    robot = gen.make_robot(torch.from_numpy(qpos))
    gt = _try_gt_simulator_pc(qpos, gen)
    left_g = _to_numpy(robot["left_geometry_pc"])
    right_g = _to_numpy(robot["right_geometry_pc"])
    left_t = _to_numpy(robot["left_topology_pc"])
    right_t = _to_numpy(robot["right_topology_pc"])
    left_world, right_world = _per_link_world_clouds(gen, qpos)

    write_gt_html(out_dir / "01_gt.html", gt)
    write_geometry_html(out_dir / "02_geometry.html", left_g, right_g)
    write_topology_html(
        out_dir / "03_topology.html",
        left_t,
        right_t,
        gen.left_links,
        gen.right_links,
        gen.left_neighbor_pairs,
        gen.right_neighbor_pairs,
        left_world,
        right_world,
    )
    write_index(out_dir / "index.html")

    print(f"Saved HTML (interactive 3D) → {out_dir}")
    print("  Open index.html in a browser (needs network for three.js CDN once).")
    print(f"  Topology overview: {len(gen.left_links)} links/arm, colored by link id.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default=str(BLOCK_DIR / "viz_out"))
    parser.add_argument("--points_per_link", type=int, default=48)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rng = np.random.RandomState(args.seed)
    qpos = rng.uniform(-0.4, 0.4, size=14).astype(np.float32)
    qpos[6] = qpos[13] = 0.4
    visualize(qpos=qpos, out_dir=args.out_dir, points_per_link=args.points_per_link)

if __name__ == "__main__":
    main()