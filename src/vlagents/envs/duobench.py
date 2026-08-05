import copy
import datetime
import json
import logging
import os
import shlex
import subprocess
import sys
from abc import ABC
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from time import sleep
from typing import Any

import gymnasium as gym
import numpy as np
from simple_slurm import Slurm
from tqdm import tqdm

from vlagents.client import RemoteAgent
from vlagents.envs.interface import EvaluatorEnv
from vlagents.policies import Act, Agent, Obs, SingleAct, SingleObs

class RCSDuoBench(EvaluatorEnv):
    INSTRUCTIONS = {}

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


EvaluatorEnv.register("duobench/ball_maze", RCSDuoBench)
EvaluatorEnv.register("duobench/bin_sort", RCSDuoBench)
EvaluatorEnv.register("duobench/block_balance", RCSDuoBench)
EvaluatorEnv.register("duobench/carry_pot", RCSDuoBench)
EvaluatorEnv.register("duobench/join_blocks", RCSDuoBench)
EvaluatorEnv.register("duobench/hinge_chest", RCSDuoBench)
EvaluatorEnv.register("duobench/pour_marbles", RCSDuoBench)
EvaluatorEnv.register("duobench/spring_door", RCSDuoBench)
EvaluatorEnv.register("duobench/transfer_cube", RCSDuoBench)
EvaluatorEnv.register("duobench/transfer_gate", RCSDuoBench)
EvaluatorEnv.register("duobench/transfer_reorient", RCSDuoBench)