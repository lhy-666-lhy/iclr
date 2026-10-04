"""PPI_rp1: identical PPI + robot state→PC→geo/topo feature tokens into scene.

Original ``space.policy.ppi.PPI`` is unchanged. Switch via Hydra:
  policy._target_=space.policy.ppi_rp1.PPI_rp1

Training hyper-params / DiffusionHead / losses match PPI; only obs encoding
gains robot tokens concatenated into scene point tokens.
"""

from __future__ import annotations

from typing import Dict, Set

import torch
import torch.nn.functional as F

from space.common.pytorch_util import dict_apply
from space.model.vision.observation_encoder_rp1 import ObservationEncoderRP1
from space.model.vision.robot_geo_topo_tokens import ROBOT_PC_KEYS
from space.policy.ppi import PPI

_ROBOT_EXTRA_KEYS: Set[str] = set(ROBOT_PC_KEYS) | {
    "robot_qpos",
    "agent_pos_raw",
    "left_link_means",
    "right_link_means",
    "robot_state_raw",
}

class PPI_rp1(PPI):
    """PPI with ObservationEncoderRP1; diffusion head & loss identical to PPI."""

    def __init__(
        self,
        noise_scheduler_cfg,
        horizon_keyframe,
        horizon_continuous,
        n_action_steps,
        n_obs_steps,
        num_inference_steps=None,
        encoder_output_dim=288,
        use_lang=False,
        pointcloud_encoder_cfg=None,
        use_decoder=False,
        what_condition="ppi",
        predict_point_flow=True,
        # --- robot token branch (PPI_rp1 only) ---
        robot_dim=128,
        robot_points_per_link=16,
        robot_use_geometry=True,
        robot_use_topology=True,
        robot_pc_backbone="pointnet",
        robot_use_type_embed=True,
        robot_allow_ee_proxy=True,
        robot_embodiment="dual_panda",
        robot_debug=False,
        **kwargs,
    ):
        super().__init__(
            noise_scheduler_cfg=noise_scheduler_cfg,
            horizon_keyframe=horizon_keyframe,
            horizon_continuous=horizon_continuous,
            n_action_steps=n_action_steps,
            n_obs_steps=n_obs_steps,
            num_inference_steps=num_inference_steps,
            encoder_output_dim=encoder_output_dim,
            use_lang=use_lang,
            pointcloud_encoder_cfg=pointcloud_encoder_cfg,
            use_decoder=use_decoder,
            what_condition=what_condition,
            predict_point_flow=predict_point_flow,
            **kwargs,
        )
        # Replace encoder only; keep DiffusionHead / schedulers / mask from PPI.
        self.obs_encoder = ObservationEncoderRP1(
            out_channel=encoder_output_dim,
            pointcloud_encoder_cfg=pointcloud_encoder_cfg,
            use_lang=use_lang,
            use_initial_pointflow=predict_point_flow,
            robot_dim=robot_dim,
            robot_points_per_link=robot_points_per_link,
            robot_use_geometry=robot_use_geometry,
            robot_use_topology=robot_use_topology,
            robot_pc_backbone=robot_pc_backbone,
            robot_use_type_embed=robot_use_type_embed,
            robot_allow_ee_proxy=robot_allow_ee_proxy,
            robot_embodiment=robot_embodiment,
            robot_debug=robot_debug,
        )

    @staticmethod
    def _split_obs(obs: Dict):
        extras = {k: obs[k] for k in list(obs.keys()) if k in _ROBOT_EXTRA_KEYS}
        base = {k: v for k, v in obs.items() if k not in _ROBOT_EXTRA_KEYS}
        return base, extras

    @staticmethod
    def _flatten_time(x: torch.Tensor, n_obs_steps: int) -> torch.Tensor:
        """Flatten leading [B, To, ...] → [B*To, ...]."""
        return x[:, :n_obs_steps, ...].reshape(-1, *x.shape[2:])

    @classmethod
    def _flatten_obs_dict(cls, nobs: Dict, n_obs_steps: int) -> Dict:
        out = {}
        for k, v in nobs.items():
            if not torch.is_tensor(v):
                out[k] = v
                continue
            # Standard PPI keys always have time dim To.
            if k in _ROBOT_EXTRA_KEYS:
                # robot_pc may be [B, N, C] (no To) or [B, To, N, C]
                if v.ndim >= 4 or (v.ndim >= 3 and v.shape[1] == n_obs_steps and k.endswith("_pc")):
                    # Heuristic: if dim1 == To, treat as time.
                    if v.shape[1] == n_obs_steps:
                        out[k] = cls._flatten_time(v, n_obs_steps)
                    else:
                        out[k] = v
                elif v.ndim >= 2 and v.shape[1] == n_obs_steps:
                    out[k] = cls._flatten_time(v, n_obs_steps)
                else:
                    out[k] = v
            else:
                out[k] = cls._flatten_time(v, n_obs_steps)
        return out

    def _prepare_this_nobs(self, obs_dict: Dict, n_obs_steps: int) -> Dict:
        """Normalize PPI obs keys; keep robot extras raw; attach robot_state_raw."""
        base_obs, extras = self._split_obs(obs_dict)
        raw_agent = base_obs["agent_pos"][:, :n_obs_steps, ...].reshape(
            -1, base_obs["agent_pos"].shape[-1]
        )
        nobs = self.normalizer.normalize(base_obs)
        nobs["point_cloud"] = nobs["point_cloud"][..., :3]
        for k, v in extras.items():
            nobs[k] = v
        this_nobs = self._flatten_obs_dict(nobs, n_obs_steps)
        this_nobs["robot_state_raw"] = raw_agent.to(
            device=this_nobs["agent_pos"].device, dtype=this_nobs["agent_pos"].dtype
        )
        return this_nobs

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Same as PPI.predict_action, but feeds robot extras into ObservationEncoderRP1."""
        this_nobs = self._prepare_this_nobs(obs_dict, self.n_obs_steps)

        # Recover B from unflat agent_pos
        B = obs_dict["agent_pos"].shape[0]
        To = self.n_obs_steps
        T = self.horizon
        Da = self.action_dim
        device = self.device
        dtype = self.dtype

        if self.predict_point_flow:
            (
                context_coord,
                context_feat,
                lang_feat,
                state_feat,
                pn_coord,
                pn_feat,
                pointflow_feat,
                pointflow_coords,
            ) = self.obs_encoder(this_nobs)
            lang_feat = lang_feat.reshape(B, To, -1)
            state_feat = state_feat.reshape(B, To, -1)
            fixed_inputs = (
                context_coord,
                context_feat,
                lang_feat,
                state_feat,
                pn_coord,
                pn_feat,
                pointflow_feat,
                pointflow_coords,
            )
            cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            fps = this_nobs["initial_point_flow"].shape[1]
            cond_point_flow = torch.zeros(
                size=(B, self.horizon_keyframe * fps, 3), device=device, dtype=dtype
            )
            cond_mask_point_flow = torch.zeros_like(cond_point_flow, dtype=torch.bool)
            nsample, npoint_flow = self.conditional_sample_diffuser_actor(
                cond_data,
                cond_mask,
                fixed_inputs,
                condition_point_flow=cond_point_flow,
                condition_mask_point_flow=cond_mask_point_flow,
            )
            naction_pred = nsample[..., :Da]
            action_pred = self.normalizer["action"].unnormalize(naction_pred)
            point_flow_pred = self.normalizer["point_flow"].unnormalize(npoint_flow)
            start = To - 1
            end = start + self.n_action_steps
            action = action_pred[:, start:end]
            return {
                "action": action,
                "action_pred": action_pred,
                "point_flow_pred": point_flow_pred,
            }

        (
            context_coord,
            context_feat,
            lang_feat,
            state_feat,
            pn_coord,
            pn_feat,
        ) = self.obs_encoder(this_nobs)
        state_feat = state_feat.reshape(B, To, -1)
        lang_feat = lang_feat.reshape(B, To, -1)
        fixed_inputs = (
            context_coord,
            context_feat,
            lang_feat,
            state_feat,
            pn_coord,
            pn_feat,
        )
        cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        nsample = self.conditional_sample_diffuser_actor(cond_data, cond_mask, fixed_inputs)
        naction_pred = nsample[..., :Da]
        action_pred = self.normalizer["action"].unnormalize(naction_pred)
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:, start:end]
        return {"action": action, "action_pred": action_pred}

    def forward(self, batch):
        """Same loss as PPI.forward; only obs→encoder path adds robot tokens."""
        base_obs, extras = self._split_obs(batch["obs"])
        raw_agent = base_obs["agent_pos"][:, : self.n_obs_steps, ...].reshape(
            -1, base_obs["agent_pos"].shape[-1]
        )

        nobs = self.normalizer.normalize(base_obs)
        nactions = self.normalizer["action"].normalize(batch["action"])
        if self.predict_point_flow:
            npoint_flow = self.normalizer["point_flow"].normalize(batch["point_flow"])
        nobs["point_cloud"] = nobs["point_cloud"][..., :3]
        for k, v in extras.items():
            nobs[k] = v

        batch_size = nactions.shape[0]
        trajectory = nactions
        cond_data = trajectory
        this_nobs = self._flatten_obs_dict(nobs, self.n_obs_steps)
        this_nobs["robot_state_raw"] = raw_agent.to(
            device=nactions.device, dtype=nactions.dtype
        )

        if self.predict_point_flow:
            point_flow_trajectory = npoint_flow[:, -self.horizon_keyframe :, :, :]
            point_flow_trajectory = point_flow_trajectory.reshape(batch_size, -1, 3)
            (
                context_coord,
                context_feat,
                lang_feat,
                state_feat,
                pn_coord,
                pn_feat,
                pointflow_feat,
                pointflow_coords,
            ) = self.obs_encoder(this_nobs)
            lang_feat = lang_feat.reshape(batch_size, self.n_obs_steps, -1)
            state_feat = state_feat.reshape(batch_size, self.n_obs_steps, -1)
            fixed_inputs = (
                context_coord,
                context_feat,
                lang_feat,
                state_feat,
                pn_coord,
                pn_feat,
                pointflow_feat,
                pointflow_coords,
            )
        else:
            (
                context_coord,
                context_feat,
                lang_feat,
                state_feat,
                pn_coord,
                pn_feat,
            ) = self.obs_encoder(this_nobs)
            lang_feat = lang_feat.reshape(batch_size, self.n_obs_steps, -1)
            state_feat = state_feat.reshape(batch_size, self.n_obs_steps, -1)
            fixed_inputs = (
                context_coord,
                context_feat,
                lang_feat,
                state_feat,
                pn_coord,
                pn_feat,
            )

        # ---- below: verbatim PPI.forward loss path ----
        condition_mask = self.mask_generator(trajectory.shape)
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        timesteps = torch.randint(
            0,
            self.noise_scheduler_cfg.num_train_timesteps,
            (bsz,),
            device=trajectory.device,
        ).long()

        gt_openess_left = trajectory[..., 14:15]
        gt_openess_right = trajectory[..., 15:16]
        pos_left = self.position_noise_scheduler.add_noise(
            trajectory[..., :3], noise[..., :3], timesteps
        )
        rot_left = self.rotation_noise_scheduler.add_noise(
            trajectory[..., 3:7], noise[..., 3:7], timesteps
        )
        pos_right = self.position_noise_scheduler.add_noise(
            trajectory[..., 7:10], noise[..., 7:10], timesteps
        )
        rot_right = self.rotation_noise_scheduler.add_noise(
            trajectory[..., 10:14], noise[..., 10:14], timesteps
        )

        noisy_trajectory = torch.cat(
            (pos_left, rot_left, pos_right, rot_right, gt_openess_left, gt_openess_right),
            -1,
        )
        noisy_trajectory[condition_mask] = cond_data[condition_mask]
        noisy_trajectory_left = noisy_trajectory[..., :7]
        noisy_trajectory_right = noisy_trajectory[..., 7:14]

        if self.predict_point_flow:
            pred_left, pred_right, pred_point_flow = self.model(
                noisy_trajectory_left,
                noisy_trajectory_right,
                timesteps,
                fixed_inputs,
            )
        else:
            pred_left, pred_right = self.model(
                noisy_trajectory_left,
                noisy_trajectory_right,
                timesteps,
                fixed_inputs,
            )

        total_loss = 0
        if self.what_condition == "pointflow_continuous":
            for layer_pred_left in pred_left:
                trans_left = layer_pred_left[:, : self.horizon_continuous, :3]
                rot_left = layer_pred_left[:, : self.horizon_continuous, 3:7]
                loss_left = 30 * F.l1_loss(
                    trans_left, noise[:, : self.horizon_continuous, :3], reduction="mean"
                ) + 10 * F.l1_loss(
                    rot_left, noise[:, : self.horizon_continuous, 3:7], reduction="mean"
                )
                if torch.numel(gt_openess_left) > 0:
                    openess_left = layer_pred_left[:, : self.horizon_continuous, 7:8]
                    loss_left += 30 * F.l1_loss(
                        openess_left,
                        gt_openess_left[:, : self.horizon_continuous, :],
                        reduction="mean",
                    )
                total_loss = total_loss + loss_left

            for layer_pred_right in pred_right:
                trans_right = layer_pred_right[:, : self.horizon_continuous, :3]
                rot_right = layer_pred_right[:, : self.horizon_continuous, 3:7]
                loss_right = 30 * F.l1_loss(
                    trans_right,
                    noise[:, : self.horizon_continuous, 7:10],
                    reduction="mean",
                ) + 10 * F.l1_loss(
                    rot_right,
                    noise[:, : self.horizon_continuous, 10:14],
                    reduction="mean",
                )
                if torch.numel(gt_openess_right) > 0:
                    openess_right = layer_pred_right[:, : self.horizon_continuous, 7:8]
                    loss_right += 30 * F.l1_loss(
                        openess_right,
                        gt_openess_right[:, : self.horizon_continuous, :],
                        reduction="mean",
                    )
                total_loss = total_loss + loss_right
            loss_dict = {"action_loss": loss_left.item() + loss_right.item()}
        else:
            for layer_pred_left in pred_left:
                trans_left = layer_pred_left[..., :3]
                rot_left = layer_pred_left[..., 3:7]
                loss_left = 30 * F.l1_loss(
                    trans_left, noise[..., :3], reduction="mean"
                ) + 10 * F.l1_loss(rot_left, noise[..., 3:7], reduction="mean")
                if torch.numel(gt_openess_left) > 0:
                    openess_left = layer_pred_left[..., 7:8]
                    loss_left += 30 * F.l1_loss(
                        openess_left, gt_openess_left, reduction="mean"
                    )
                total_loss = total_loss + loss_left

            for layer_pred_right in pred_right:
                trans_right = layer_pred_right[..., :3]
                rot_right = layer_pred_right[..., 3:7]
                loss_right = 30 * F.l1_loss(
                    trans_right, noise[..., 7:10], reduction="mean"
                ) + 10 * F.l1_loss(rot_right, noise[..., 10:14], reduction="mean")
                if torch.numel(gt_openess_right) > 0:
                    openess_right = layer_pred_right[..., 7:8]
                    loss_right += 30 * F.l1_loss(
                        openess_right, gt_openess_right, reduction="mean"
                    )
                total_loss = total_loss + loss_right
            loss_dict = {"action_loss": loss_left.item() + loss_right.item()}

        if self.predict_point_flow:
            for layer_pred_point_flow in pred_point_flow:
                trans_point_flow = layer_pred_point_flow[..., :3]
                loss_point_flow = 600 * F.l1_loss(
                    trans_point_flow, point_flow_trajectory[..., :3], reduction="mean"
                )
                total_loss = total_loss + loss_point_flow
            loss_dict["point_flow_loss"] = loss_point_flow.item()
        loss_dict["bc_loss"] = total_loss.item()
        return total_loss, loss_dict
