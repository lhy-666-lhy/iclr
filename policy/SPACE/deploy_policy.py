# import packages and module here
import sys

import torch
import sapien.core as sapien
import traceback
import os
import numpy as np
from envs import *
from hydra import initialize, compose
from omegaconf import OmegaConf
from hydra.core.hydra_config import HydraConfig
from hydra import main as hydra_main
import pathlib
from omegaconf import OmegaConf

import yaml
from datetime import datetime
import importlib

from hydra import initialize, compose
from omegaconf import OmegaConf
from datetime import datetime

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)

sys.path.append(os.path.join(parent_directory, 'spacepolicy'))

from space_policy import *

def encode_obs(observation):  # Post-Process Observation
    obs = dict()
    obs['agent_pos'] = observation['joint_action']['vector']
    obs['point_cloud'] = observation['pointcloud']
    return obs

def get_model(usr_args):
    config_path = "./spacepolicy/backbone/config"
    config_name = f"{usr_args['config_name']}.yaml"

    with initialize(config_path=config_path, version_base='1.2'):
        cfg = compose(config_name=config_name)

    now = datetime.now()
    run_dir = f"data/outputs/{now:%Y.%m.%d}/{now:%H.%M.%S}_{usr_args['config_name']}_{usr_args['task_name']}"

    hydra_runtime_cfg = {
        "job": {
            "override_dirname": usr_args['task_name']
        },
        "run": {
            "dir": run_dir
        },
        "sweep": {
            "dir": run_dir,
            "subdir": "0"
        }
    }

    OmegaConf.set_struct(cfg, False)
    cfg.hydra = hydra_runtime_cfg
    cfg.task_name = usr_args["task_name"]
    cfg.expert_data_num = usr_args["expert_data_num"]
    cfg.raw_task_name = usr_args["task_name"]
    cfg.policy.use_pc_color = usr_args['use_rgb']

    # Eval builds the matching internal structure from config (IO is shared).
    if usr_args.get("use_robot_aware") is not None:
        v = usr_args["use_robot_aware"]
        if isinstance(v, str):
            cfg.policy.use_robot_aware = v.strip().lower() in ("1", "true", "yes", "y")
        else:
            cfg.policy.use_robot_aware = bool(v)
    encoder_type = usr_args.get("robot_aware_encoder_type", None)
    if encoder_type is not None:
        # eval_policy.py does eval(override_value); bare "main" becomes function main().
        if callable(encoder_type):
            encoder_type = getattr(encoder_type, "__name__", "main")
        cfg.policy.robot_aware_encoder_type = str(encoder_type)
    # Match train.sh HAND_H / INJ_H so eval architecture aligns with ckpt.
    if os.environ.get("HAND_H"):
        cfg.policy.robot_aware_hand_hidden_dim = int(os.environ["HAND_H"])
    if os.environ.get("INJ_H"):
        cfg.policy.robot_aware_injection_hidden_dim = int(os.environ["INJ_H"])
    print(
        f"[eval get_model] use_robot_aware={cfg.policy.use_robot_aware} "
        f"robot_aware_encoder_type={cfg.policy.robot_aware_encoder_type} "
        f"hand_h={cfg.policy.robot_aware_hand_hidden_dim} "
        f"inj_h={cfg.policy.robot_aware_injection_hidden_dim}",
        flush=True,
    )
    OmegaConf.set_struct(cfg, True)

    SPACE_Model = SPACE(cfg, usr_args)
    return SPACE_Model

def eval(TASK_ENV, model, observation):
    obs = encode_obs(observation)  # Post-Process Observation
    # instruction = TASK_ENV.get_instruction()

    if len(
            model.env_runner.obs
    ) == 0:  # Force an update of the observation at the first frame to avoid an empty observation window, `obs_cache` here can be modified
        model.update_obs(obs)

    actions = model.get_action()  # Get Action according to observation chunk

    for action in actions:  # Execute each step of the action
        TASK_ENV.take_action(action)
        observation = TASK_ENV.get_obs()
        obs = encode_obs(observation)
        model.update_obs(obs)  # Update Observation, `update_obs` here can be modified

def reset_model(
        model):  # Clean the model cache at the beginning of every evaluation episode, such as the observation window
    model.env_runner.reset_obs()
