"""Observation encoder for PPI_rp1 = original PPI encoder + robot geo/topo tokens."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from space.model.vision.observation_encoder import ObservationEncoder
from space.model.vision.robot_geo_topo_tokens import RobotGeoTopoTokenEncoder

class ObservationEncoderRP1(ObservationEncoder):
    """Run original ObservationEncoder, then concat robot tokens into scene pcd."""

    def __init__(
        self,
        out_channel=288,
        state_mlp_size=(128, 288),
        state_mlp_activation_fn=nn.ReLU,
        lang_mlp_size=(288, 288),
        lang_mlp_activation_fn=nn.ReLU,
        pcd_mlp_size=(288, 288),
        pcd_mlp_activation_fn=nn.ReLU,
        pointcloud_encoder_cfg=None,
        use_lang=False,
        use_initial_pointflow=True,
        scene_pcd_num=6144,
        robot_dim=128,
        robot_points_per_link=16,
        robot_use_geometry=True,
        robot_use_topology=True,
        robot_pc_backbone="pointnet",
        robot_use_type_embed=True,
        robot_allow_ee_proxy=True,
        robot_embodiment="dual_panda",
        robot_debug=False,
    ):
        super().__init__(
            out_channel=out_channel,
            state_mlp_size=state_mlp_size,
            state_mlp_activation_fn=state_mlp_activation_fn,
            lang_mlp_size=lang_mlp_size,
            lang_mlp_activation_fn=lang_mlp_activation_fn,
            pcd_mlp_size=pcd_mlp_size,
            pcd_mlp_activation_fn=pcd_mlp_activation_fn,
            pointcloud_encoder_cfg=pointcloud_encoder_cfg,
            use_lang=use_lang,
            use_initial_pointflow=use_initial_pointflow,
            scene_pcd_num=scene_pcd_num,
        )
        self.robot_token_encoder = RobotGeoTopoTokenEncoder(
            token_dim=out_channel,
            robot_dim=robot_dim,
            points_per_link=robot_points_per_link,
            use_geometry=robot_use_geometry,
            use_topology=robot_use_topology,
            pc_backbone=robot_pc_backbone,
            use_type_embed=robot_use_type_embed,
            allow_ee_proxy=robot_allow_ee_proxy,
            robot_embodiment=robot_embodiment,
            debug=robot_debug,
        )

    def forward(self, observations: Dict):
        base = super().forward(observations)
        robot_feat, robot_coord = self.robot_token_encoder(observations)

        if self.use_initial_pointflow:
            (
                points,
                pcd_feat,
                lang_feat,
                state_feat,
                sampled_pcd_coord,
                sampled_pcd_feat,
                point_flow_feat,
                pointflow_coords,
            ) = base
            points = torch.cat([points, robot_coord], dim=1)
            pcd_feat = torch.cat([pcd_feat, robot_feat], dim=1)
            return (
                points,
                pcd_feat,
                lang_feat,
                state_feat,
                sampled_pcd_coord,
                sampled_pcd_feat,
                point_flow_feat,
                pointflow_coords,
            )

        (
            points,
            pcd_feat,
            lang_feat,
            state_feat,
            sampled_pcd_coord,
            sampled_pcd_feat,
        ) = base
        points = torch.cat([points, robot_coord], dim=1)
        pcd_feat = torch.cat([pcd_feat, robot_feat], dim=1)
        return (
            points,
            pcd_feat,
            lang_feat,
            state_feat,
            sampled_pcd_coord,
            sampled_pcd_feat,
        )
