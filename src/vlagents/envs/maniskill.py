import copy
import importlib
import logging
from typing import Any, ClassVar

import gymnasium as gym
import numpy as np

from vlagents.envs.interface import EvalEnv
from vlagents.policies.interface import Obs, SingleAct, SingleObs


class HumanCameraWrapper(gym.ObservationWrapper):
    """
    Flattens the rgbd mode observations into a dictionary with two keys, "rgbd" and "state"

    Args:
        rgb (bool): Whether to include rgb images in the observation
        depth (bool): Whether to include depth images in the observation
        state (bool): Whether to include state data in the observation

    Note that the returned observations will have a "rgbd" or "rgb" or "depth" key depending on the rgb/depth bool flags.
    """

    def __init__(self, env) -> None:
        self.base_env = env.unwrapped
        super().__init__(env)
        new_obs = self.observation(self.base_env._init_raw_obs)
        self.base_env.update_obs_space(new_obs)

    def observation(self, observation: dict):
        # ret = dict()
        if not hasattr(self.env, "_has_reset") or not self.env._has_reset:
            self.env.reset()
            self.env._has_reset = True
        # observation["sensor_data"]["human_camera"] = dict(rgb=self.env.render())
        observation["sensor_data"]["base_camera"] = dict(rgb=self.env.render())
        return observation


class ManiSkill(EvalEnv):
    INSTRUCTIONS: ClassVar[dict[str, str]] = {
        "LiftPegUpright-v1": "lift the peg upright",
        "PegInsertionSide-v1": "insert the peg from the side",
        "PickCube-v1": "pick up the cube",
        "PlugCharger-v1": "plug the charger in",
        "PullCube-v1": "pull the cube towards the robot base",
        "PullCubeTool-v1": "pull the cube by using the red tool",
        "PushCube-v1": "push the cube away from the robot base",
        "PushT-v1": "align the T shape",
        "RollBall-v1": "push the ball",
        "StackCube-v1": "stack the red cube on the green cube",
        "PokeCube-v1": "push the cube by using the blue tool",
    }

    def __init__(self, env_id, **env_kwargs):
        # TODO: one could save only every nth episode by adding an episode counter which steps the record env only
        # when the counter is divisible by n otherwise steps the normal env
        logging.info(f"Creating ManiSkill env {env_id}")
        output_dir = env_kwargs.pop("video_dir", None)
        super().__init__(env_id, **env_kwargs)
        logging.info(f"Created ManiSkill env {env_id}")
        if "human_render_camera_configs" in env_kwargs:
            self.env = HumanCameraWrapper(self.env)

        if output_dir is not None:
            logging.info(f"Recording to {output_dir}")
            from mani_skill.utils import wrappers

            self.env = wrappers.RecordEpisode(
                self.env,
                output_dir,
                save_on_reset=True,
                save_trajectory=True,
                trajectory_name=f"eval-{env_id}",
                save_video=True,
                video_fps=30,
                record_reward=True,
            )
        logging.info(f"Done Created ManiSkill env {env_id}")

    def translate_obs(self, obs: dict[str, Any]) -> Obs:
        # does not include history
        return Obs(
            obs={
                "default": SingleObs(cameras={"rgb_side": obs["sensor_data"]["base_camera"]["rgb"].squeeze(0).numpy()})
            },
            language_instruction=self.language_instruction,
        )

    def step(self, action: dict[str, SingleAct]) -> tuple[Obs, float, bool, bool, dict]:
        # includes horizon
        # careful with gripper action: the model needs to be trained on [-1, 1] interval
        assert len(action) == 1, "ManiSkill expects a single robot action"
        _, robot_action = next(iter(action.items()))
        a = np.asarray(copy.copy(robot_action.action), dtype=np.float32)
        if robot_action.gripper is not None:
            a = np.concatenate([a, np.asarray([robot_action.gripper], dtype=np.float32)])
        if self.env_id == "PushT-v1":
            if robot_action.gripper is not None:
                a = a[:-1]
        else:
            a[-1] = a[-1] * 2 - 1.0
        obs, reward, success, truncated, info = self.env.step(a)
        return self.translate_obs(obs), reward, success, truncated, info

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Obs, dict[str, Any]]:
        # maniskill has a bug that does not allow None in options
        obs, info = self.env.reset(seed=seed)
        return self.translate_obs(obs), info

    @property
    def language_instruction(self) -> str:
        return self.INSTRUCTIONS[self.env_id]

    @staticmethod
    def do_import():
        import mani_skill.envs


EvalEnv.register("LiftPegUpright-v1", ManiSkill)
EvalEnv.register("PegInsertionSide-v1", ManiSkill)
EvalEnv.register("PickCube-v1", ManiSkill)
EvalEnv.register("PlugCharger-v1", ManiSkill)
EvalEnv.register("PullCube-v1", ManiSkill)
EvalEnv.register("PullCubeTool-v1", ManiSkill)
EvalEnv.register("PushCube-v1", ManiSkill)
EvalEnv.register("PushT-v1", ManiSkill)
EvalEnv.register("RollBall-v1", ManiSkill)
EvalEnv.register("StackCube-v1", ManiSkill)
EvalEnv.register("PokeCube-v1", ManiSkill)
