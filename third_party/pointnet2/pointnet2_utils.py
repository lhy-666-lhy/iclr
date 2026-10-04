"""PointNet++ pure-PyTorch ops tuned for small robot point clouds.

Replaces slow ball-query ([B,S,N] index + full sort) with kNN topk.
FPS npoint is clamped to N; designed for N≈16–144 (not ModelNet 1024).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

def square_distance(src, dst):
    """src [B,N,C], dst [B,M,C] → [B,N,M] squared distances."""
    return (
        (src**2).sum(-1, keepdim=True)
        + (dst**2).sum(-1).unsqueeze(1)
        - 2.0 * torch.matmul(src, dst.transpose(1, 2))
    )

def index_points(points, idx):
    """points [B,N,C], idx [B,...] → gather along point dim."""
    B = points.shape[0]
    bi = torch.arange(B, device=points.device)
    if idx.ndim == 2:
        return points[bi[:, None], idx]
    if idx.ndim == 3:
        return points[bi[:, None, None], idx]
    raise ValueError(f"idx ndim must be 2 or 3, got {idx.ndim}")

def farthest_point_sample(xyz, npoint):
    """Iterative FPS. npoint is small (≤32) for robot PCs → cheap enough."""
    device = xyz.device
    B, N, _ = xyz.shape
    npoint = int(min(max(int(npoint), 1), N))
    if npoint == N:
        return torch.arange(N, device=device, dtype=torch.long).view(1, N).expand(B, N)

    centroids = torch.empty(B, npoint, dtype=torch.long, device=device)
    distance = torch.full((B, N), 1e10, device=device, dtype=xyz.dtype)
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_indices = torch.arange(B, dtype=torch.long, device=device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].unsqueeze(1)
        dist = ((xyz - centroid) ** 2).sum(-1)
        distance = torch.minimum(distance, dist)
        farthest = distance.argmax(dim=-1)
    return centroids

def knn_point(nsample, xyz, new_xyz):
    """kNN indices of ``xyz`` for each query in ``new_xyz``. [B,S,k]."""
    N = xyz.shape[1]
    k = int(min(max(int(nsample), 1), N))
    dist = square_distance(new_xyz, xyz)
    _, idx = dist.topk(k, dim=-1, largest=False, sorted=False)
    return idx

def sample_and_group(npoint, radius, nsample, xyz, points):
    """FPS + kNN group (radius kept for API compat, unused)."""
    del radius  # kNN grouping is faster/stabler on tiny robot clouds
    B, N, C = xyz.shape
    npoint = int(min(max(int(npoint), 1), N))
    fps_idx = farthest_point_sample(xyz, npoint)
    new_xyz = index_points(xyz, fps_idx)
    idx = knn_point(nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx)
    grouped_xyz_norm = grouped_xyz - new_xyz.unsqueeze(2)
    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
    else:
        new_points = grouped_xyz_norm
    return new_xyz, new_points

def sample_and_group_all(xyz, points):
    B, N, C = xyz.shape
    new_xyz = xyz.new_zeros(B, 1, C)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points

class PointNetSetAbstraction(nn.Module):
    """One SA level. Uses Conv1d-equivalent Conv2d + LayerNorm-friendly BN."""

    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all: bool):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last = in_channel
        for out_c in mlp:
            self.mlp_convs.append(nn.Conv2d(last, out_c, 1))
            # GroupNorm is stabler than BN when B*K is huge but spatial is tiny
            self.mlp_bns.append(nn.GroupNorm(1, out_c))
            last = out_c

    def forward(self, xyz, points):
        # xyz/points: [B,C,N] channel-first (PointNet++ convention)
        xyz = xyz.permute(0, 2, 1).contiguous()
        if points is not None:
            points = points.permute(0, 2, 1).contiguous()
        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(
                self.npoint, self.radius, self.nsample, xyz, points
            )
        # [B,npoint,k,C] → [B,C,k,npoint]
        new_points = new_points.permute(0, 3, 2, 1).contiguous()
        for conv, bn in zip(self.mlp_convs, self.mlp_bns):
            new_points = F.relu(bn(conv(new_points)), inplace=True)
        new_points = new_points.max(dim=2)[0]
        new_xyz = new_xyz.permute(0, 2, 1).contiguous()
        return new_xyz, new_points
