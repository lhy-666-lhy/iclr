if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import hydra
from omegaconf import OmegaConf
import pathlib
import sys
from train_space import TrainSPACEWorkspace

OmegaConf.register_new_resolver("eval", eval, replace=True)

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath("backbone", "config")),
)
def main(cfg):
    workspace = TrainSPACEWorkspace(cfg)
    workspace.eval()

class SPACE:

    def __init__(self, cfg, usr_args) -> None:
        self.policy, self.env_runner = self.get_policy_and_runner(cfg, usr_args)

    def update_obs(self, observation):
        self.env_runner.update_obs(observation)

    def get_action(self, observation=None):
        action = self.env_runner.get_action(self.policy, observation)
        return action

    def get_policy_and_runner(self, cfg, usr_args):
        workspace = TrainSPACEWorkspace(cfg)
        policy, env_runner = workspace.get_policy_and_runner(cfg, usr_args)
        return policy, env_runner

if __name__ == "__main__":
    main()
