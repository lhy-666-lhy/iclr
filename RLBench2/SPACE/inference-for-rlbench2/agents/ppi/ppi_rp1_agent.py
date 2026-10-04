"""PPI_rp1 eval agent: same as PPIAgent + live full-link DualPanda robot_pc each step."""

from __future__ import annotations

import torch

from agents.ppi.ppi_agent import PPIAgent
from space.utils.dual_panda_robot_pc import ROBOT_PC_NPZ_KEYS, sample_dual_panda_robot_pc_live

class PPI_rp1Agent(PPIAgent):
    def __init__(self, *args, robot_points_per_link: int = 16, **kwargs):
        super().__init__(*args, **kwargs)
        self.robot_points_per_link = int(robot_points_per_link)

    def _extra_actor_obs(self, observation: dict) -> dict:
        # Fresh full-link mesh PC from current PyRep scene every act() step.
        pc = sample_dual_panda_robot_pc_live(
            points_per_link=self.robot_points_per_link,
            seed=0,
        )
        out = {}
        for key in ROBOT_PC_NPZ_KEYS:
            t = torch.from_numpy(pc[key]).to(device=self._device, dtype=torch.float32)
            out[key] = t.unsqueeze(0)
        return out
