# PPI_rp1 eval agent factory (aligned with launch_utils.py, model + robot_pc differ).

from helpers.preprocess_agent import PreprocessAgent
from space.policy.ppi_rp1 import PPI_rp1
from agents.ppi.ppi_rp1_agent import PPI_rp1Agent
from omegaconf import DictConfig

def create_agent(cfg: DictConfig):
    pol = cfg.method.policy
    actor_net = PPI_rp1(
        noise_scheduler_cfg=pol.noise_scheduler_cfg,
        horizon_keyframe=pol.horizon_keyframe,
        horizon_continuous=pol.horizon_continuous,
        n_action_steps=pol.n_action_steps,
        n_obs_steps=pol.n_obs_steps,
        num_inference_steps=pol.num_inference_steps,
        encoder_output_dim=pol.encoder_output_dim,
        use_lang=pol.use_lang,
        pointcloud_encoder_cfg=pol.pointcloud_encoder_cfg,
        use_decoder=False,
        what_condition=pol.what_condition,
        predict_point_flow=pol.predict_point_flow,
        robot_embodiment=getattr(pol, "robot_embodiment", "dual_panda"),
        robot_dim=getattr(pol, "robot_dim", 128),
        robot_points_per_link=getattr(pol, "robot_points_per_link", 16),
        robot_use_geometry=getattr(pol, "robot_use_geometry", True),
        robot_use_topology=getattr(pol, "robot_use_topology", True),
        robot_pc_backbone=getattr(pol, "robot_pc_backbone", "pointnet"),
        robot_use_type_embed=getattr(pol, "robot_use_type_embed", True),
        robot_allow_ee_proxy=False,
        robot_debug=False,
    )

    ppi_agent = PPI_rp1Agent(
        actor_network=actor_net,
        cameras=cfg.rlbench.cameras,
        task_name=cfg.rlbench.tasks[0],
        weight_name=cfg.framework.weight_name,
        fps_num=pol.fps_num,
        cameras_pcd=cfg.rlbench.cameras_pcd,
        use_pc_color=pol.use_pc_color,
        use_lang=pol.use_lang,
        bounding_box=pol.bounding_box,
        episode_length=cfg.rlbench.episode_length,
        prediction_type=pol.prediction_type,
        save_path=cfg.cinematic_recorder.save_path,
        query_freq=cfg.rlbench.query_freq,
        horizon_continuous=pol.horizon_continuous,
        horizon_keyframe=pol.horizon_keyframe,
        predict_point_flow=pol.predict_point_flow,
        pointflow_num=pol.pointflow_num,
        text_prompt=pol.text_prompt,
        prompt_type=pol.prompt_type,
        sample_type=pol.sample_type,
        sam_cameras=pol.sam_cameras,
        ckpt_name=cfg.framework.ckpt_name,
        jump_step=cfg.framework.jump_step,
        sam_checkpoint_path=pol.sam_checkpoint_path,
        gdino_config_path=pol.gdino_config_path,
        gdino_checkpoint_path=pol.gdino_checkpoint_path,
        bert_model_path=getattr(pol, "bert_model_path", None),
        instruction_embeddings_path=pol.instruction_embeddings_path,
        robot_points_per_link=getattr(pol, "robot_points_per_link", 16),
    )

    return PreprocessAgent(pose_agent=ppi_agent, norm_rgb=False)
