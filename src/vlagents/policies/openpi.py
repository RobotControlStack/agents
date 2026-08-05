import logging

import numpy as np
from vlagents import register_agent
from vlagents.policies.interface import Act, Agent, Obs

class OpenPiModel(Agent):
    def __init__(
        self,
        train_config_name: str = "pi0_droid",
        default_checkpoint_path: str = "gs://openpi-assets/checkpoints/pi0_droid",
        **kwargs,
    ) -> None:
        super().__init__(default_checkpoint_path=default_checkpoint_path, **kwargs)
        from openpi.training import config

        logging.info(f"checkpoint_path: {self.checkpoint_path}, checkpoint_step: {self.checkpoint_step}")
        self.openpi_path = self.checkpoint_path.format(checkpoint_step=self.checkpoint_step)

        self.cfg = config.get_config(train_config_name)

    def initialize(self):
        from openpi.policies import policy_config
        from openpi.shared import download

        checkpoint_dir = download.maybe_download(self.openpi_path)

        # Create a trained policy.
        self.policy = policy_config.create_trained_policy(self.cfg, checkpoint_dir)

    def act(self, obs: Obs) -> Act:
        super().act(obs)
        robot_name, single_obs = self._require_single_arm(obs)
        observation = {
            # OpenPI expects channel-first images: [H, W, C] -> [C, H, W]
            f"observation/{k}": np.copy(v).transpose(2, 0, 1)
            for k, v in single_obs.cameras.items()
        }
        observation.update(
            {
                # openpi expects 0 as gripper open and 1 as closed
                "observation/state": np.concatenate(
                    [np.asarray(single_obs.joints, dtype=np.float32), [1 - float(single_obs.gripper or 0.0)]]
                ),
                "prompt": obs.language_instruction,
            }
        )
        action_chunk = np.asarray(self.policy.infer(observation)["actions"], dtype=np.float32)
        action_chunk[:, -1] = 1 - action_chunk[:, -1]
        return self._chunk_act(robot_name, action_chunk[:, :-1], grippers=action_chunk[:, -1])

register_agent("openpi", OpenPiModel)