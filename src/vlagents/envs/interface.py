import logging
from abc import ABC
from dataclasses import dataclass
from typing import Any

import gymnasium as gym

from vlagents.policies.interface import Act, Obs, SingleAct

logging.basicConfig(
    format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)


class EvalEnv(ABC):
    ENVS: dict[str, "EvalEnv"] = {}

    def __init__(self, env_id: str, execution_horizon: int | None = None, **env_kwargs) -> None:
        self.do_import()
        self.env = gym.make(env_id, **env_kwargs)
        self.env_id = env_id
        self.execution_horizon = execution_horizon
        self.last_chunk_steps = 0

    def chunk_step(self, actions: Act, max_steps: int | None = None) -> tuple[Obs, float, bool, bool, dict[str, Any]]:
        if not actions.acts:
            raise ValueError("Agents must return at least one action")
        rewards = []
        self.last_chunk_steps = 0
        for action in actions.acts:
            if max_steps is not None and self.last_chunk_steps >= max_steps:
                break
            obs, reward, done, truncated, info = self.step(action)
            rewards.append(reward)
            self.last_chunk_steps += 1
            if (
                done
                or truncated
                or (self.execution_horizon is not None and self.last_chunk_steps >= self.execution_horizon)
            ):
                break
        if not rewards:
            raise ValueError("max_steps must allow at least one environment step")
        return obs, sum(rewards), done, truncated, info

    def step(self, action: dict[str, SingleAct]) -> tuple[Obs, float, bool, bool, dict]:
        raise NotImplementedError

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Obs, dict[str, Any]]:
        raise NotImplementedError

    @property
    def language_instruction(self) -> str:
        raise NotImplementedError

    @staticmethod
    def register(env_id: str, env: "EvalEnv") -> None:
        EvalEnv.ENVS[env_id] = env

    @staticmethod
    def make(env_id: str, **env_kwargs) -> "EvalEnv":
        return EvalEnv.ENVS[env_id](env_id, **env_kwargs)

    @staticmethod
    def do_import():
        raise NotImplementedError



@dataclass
class EvalConfig:
    env_id: str
    env_kwargs: dict[str, Any]
    execution_horizon: int | None = None
    max_steps_per_episode: int = 100
    seed: int = 42
    same_machine: bool = False
    jpeg_encoding: bool = False
    image_size: tuple[int, int] | None = (224, 224)


@dataclass
class AgentConfig:
    host: str
    agent_name: str
    agent_kwargs: dict[str, Any]
    python_path: str = "python"
    """modify this if you want to use a specific python environment """
    port: int = 8080

