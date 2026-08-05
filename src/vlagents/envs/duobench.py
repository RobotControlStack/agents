from typing import Any, ClassVar

import numpy as np

from vlagents import register_env
from vlagents.envs.interface import EvalEnv
from vlagents.policies.interface import Obs, SingleAct, SingleObs


class RCSDuoBench(EvalEnv):
    INSTRUCTIONS: ClassVar[dict[str, str]] = {}

    def __init__(self, env_id, **env_kwargs):
        self.robot_keys: str = env_kwargs.pop("robot_keys", ["left", "right"])
        self.control_mode: str = env_kwargs.pop("control_mode", "joints")
        self._instruction: str | None = None
        super().__init__(env_id, **env_kwargs)

    def translate_obs(self, obs: dict[str, Any]) -> Obs:
        cameras = {key: obs["frames"][key]["rgb"]["data"] for key in obs["frames"]}
        return Obs(
            obs={
                robot_key: SingleObs(
                    cameras=cameras.copy(),
                    joints=np.asarray(obs[robot_key]["joints"], dtype=np.float32),
                    gripper=float(obs[robot_key]["gripper"]),
                )
                for robot_key in self.robot_keys
            },
            language_instruction=self.language_instruction,
        )

    def step(self, action: dict[str, SingleAct]) -> tuple[Obs, float, bool, bool, dict]:
        env_action = {}
        for robot in self.robot_keys:
            robot_action = action[robot]
            gripper = 0.0 if robot_action.gripper is None else robot_action.gripper
            if self.control_mode == "joints":
                env_action[robot] = {
                    "joints": np.asarray(robot_action.action, dtype=np.float32),
                    "gripper": np.asarray([gripper], dtype=np.float32),
                }
            else:
                env_action[robot] = {
                    "xyzrpy": np.asarray(robot_action.action, dtype=np.float32),
                    "gripper": np.asarray([gripper], dtype=np.float32),
                }
        obs, reward, success, truncated, info = self.env.step(env_action)
        r = float(reward)

        return self.translate_obs(obs), r, success, truncated, info

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Obs, dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)
        self._instruction = info["instruction"]
        return self.translate_obs(obs), info

    @property
    def language_instruction(self) -> str:
        assert self._instruction is not None
        return self._instruction

    @staticmethod
    def do_import():
        import rcs
        from rcs_duobench.tasks import (
            ball_maze,
            bin_sort,
            block_balance,
            carry_pot,
            hinge_chest,
            join_blocks,
            pour_marbles,
            spring_door,
            transfer_cube,
            transfer_gate,
            transfer_reorient,
        )


register_env("duobench/ball_maze", RCSDuoBench)
register_env("duobench/bin_sort", RCSDuoBench)
register_env("duobench/block_balance", RCSDuoBench)
register_env("duobench/carry_pot", RCSDuoBench)
register_env("duobench/join_blocks", RCSDuoBench)
register_env("duobench/hinge_chest", RCSDuoBench)
register_env("duobench/pour_marbles", RCSDuoBench)
register_env("duobench/spring_door", RCSDuoBench)
register_env("duobench/transfer_cube", RCSDuoBench)
register_env("duobench/transfer_gate", RCSDuoBench)
register_env("duobench/transfer_reorient", RCSDuoBench)
