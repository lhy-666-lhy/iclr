import sys, os

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)
sys.path.append(os.path.join(parent_directory, '..'))
sys.path.append(os.path.join(parent_directory, '../..'))

from typing import Dict, List, Optional
import torch
import numpy as np
import copy
from pathlib import Path
from backbone.common.pytorch_util import dict_apply
from backbone.common.replay_buffer import ReplayBuffer
from backbone.common.sampler import (
    SequenceSampler,
    get_val_mask,
    downsample_mask,
)
from backbone.model.common.normalizer import (
    LinearNormalizer,
    SingleFieldLinearNormalizer,
)
from backbone.dataset.base_dataset import BaseDataset

# block/ helpers for encoder_type → zarr key mapping
_BLOCK_DIR = Path(__file__).resolve().parents[3] / "block"
if str(_BLOCK_DIR) not in sys.path:
    sys.path.insert(0, str(_BLOCK_DIR))
from robot_aware_encoder import (  # noqa: E402
    resolve_robot_pc_sampling,
    robot_pc_zarr_keys,
)

class RobotDataset(BaseDataset):

    def __init__(
        self,
        zarr_path,
        horizon=1,
        pad_before=0,
        pad_after=0,
        seed=42,
        val_ratio=0.0,
        max_train_episodes=None,
        task_name=None,
        use_robot_pc_cache: bool = True,
        robot_aware_encoder_type: str = "main",
        robot_aware_points_per_link: int = 16,
    ):
        super().__init__()
        self.task_name = task_name
        current_file_path = os.path.abspath(__file__)
        parent_directory = os.path.dirname(current_file_path)
        zarr_path = os.path.join(parent_directory, zarr_path)

        load_keys: List[str] = ["state", "action", "point_cloud"]
        self.robot_pc_key_map: Optional[Dict[str, str]] = None
        self.use_robot_pc_cache = bool(use_robot_pc_cache)
        if self.use_robot_pc_cache:
            region, m = resolve_robot_pc_sampling(
                robot_aware_encoder_type, robot_aware_points_per_link
            )
            key_map = robot_pc_zarr_keys(region, m)
            # Only enable cache if all keys exist in zarr
            try:
                import zarr

                root = zarr.open(zarr_path, mode="r")
                if all(k in root["data"] for k in key_map.values()):
                    self.robot_pc_key_map = key_map
                    load_keys.extend(key_map.values())
                    print(
                        f"[RobotDataset] robot PC cache ON  {region}_m{m} "
                        f"encoder={robot_aware_encoder_type}"
                    )
                else:
                    missing = [k for k in key_map.values() if k not in root["data"]]
                    print(
                        f"[RobotDataset] robot PC cache OFF (missing {missing}); "
                        f"will use online FK"
                    )
            except Exception as e:
                print(f"[RobotDataset] robot PC cache OFF ({e}); will use online FK")

        self.replay_buffer = ReplayBuffer.copy_from_path(zarr_path, keys=load_keys)
        val_mask = get_val_mask(n_episodes=self.replay_buffer.n_episodes, val_ratio=val_ratio, seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(mask=train_mask, max_n=max_train_episodes, seed=seed)
        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
        )
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask,
        )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode="limits", **kwargs):
        data = {
            "action": self.replay_buffer["action"],
            "agent_pos": self.replay_buffer["state"][..., :],
            "point_cloud": self.replay_buffer["point_cloud"],
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        agent_pos = sample["state"][
            :,
        ].astype(np.float32)  # (agent_posx2, block_posex3)
        point_cloud = sample["point_cloud"][
            :,
        ].astype(np.float32)  # (T, 1024, 6)

        data = {
            "obs": {
                "point_cloud": point_cloud,  # T, 1024, 6
                "agent_pos": agent_pos,  # T, D_pos
            },
            "action": sample["action"].astype(np.float32),  # T, D_action
        }
        if self.robot_pc_key_map is not None:
            robot_pc = {}
            for logical, zkey in self.robot_pc_key_map.items():
                robot_pc[logical] = sample[zkey].astype(np.float32)
            data["robot_pc"] = robot_pc
        return data

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data
