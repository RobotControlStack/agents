import logging
import os
from typing import Any

import numpy as np

from vlagents import register_env
from vlagents.envs.interface import EvalEnv
from vlagents.policies.interface import Obs, SingleAct, SingleObs


class Libero(EvalEnv):
    def __init__(self, env_id: str, reset_steps: int = 14, execution_horizon: int | None = None, **env_kwargs) -> None:
        """
        For supported env_kwargs checkout ControlEnv class in libero.
        We add the following env_kwargs on top:
        - task_id (int): libero task id for given task suite. The number of tasks per task suite can checked with Libero.n_tasks(env_id). Defaults to 0.
        - control_mode (str): either 'relative' or 'absolute'. Defaults to 'relative'.

        """
        logging.info("Creating Libero env")
        self.reset_steps = reset_steps
        self.control_mode = env_kwargs.pop("control_mode", "relative")
        super().__init__(env_id, execution_horizon=execution_horizon, **env_kwargs)

    @staticmethod
    def n_tasks(env_id: str) -> int:
        from libero.libero import benchmark

        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict[env_id]()
        return task_suite.n_tasks

    @staticmethod
    def _make_gym(env_id, **env_kwargs):
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv

        benchmark_dict = benchmark.get_benchmark_dict()

        task_suite = benchmark_dict[env_id]()
        task_id = min(max(env_kwargs.pop("task_id", 0), 0), task_suite.n_tasks - 1)
        task = task_suite.get_task(task_id)

        task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        env = OffScreenRenderEnv(
            bddl_file_name=task_bddl_file,
            **env_kwargs,
        )

        return env, task.language, task.name, task_suite, task_id, task

    def make_gym(self):
        (
            env,
            self._language_instruction,
            self.task_name,
            self.task_suite,
            self.task_id,
            self.task,
        ) = self._make_gym(self.env_id, **self.env_kwargs)
        logging.info(
            f"Created Libero env, task suite: {self.env_id}, task id: {self.task_id}, task name {self.task_name}, instruction: {self._language_instruction}"
        )
        return env

    def do_import(self):
        # _make_gym imports LIBERO lazily so importing vlagents does not require it.
        pass

    def translate_obs(self, obs: dict[str, Any]) -> Obs:
        joints = None
        if "robot0_joint_pos" in obs:
            joints = np.asarray(obs["robot0_joint_pos"], dtype=np.float32)
        gripper = float(np.asarray(obs["robot0_gripper_qpos"]).reshape(-1)[0] / 0.04)
        return Obs(
            obs={
                "robot0": SingleObs(
                    cameras={
                        "rgb_side": obs["agentview_image"][::-1],
                        "rgb_wrist": obs["robot0_eye_in_hand_image"][::-1],
                    },
                    joints=joints,
                    gripper=gripper,
                )
            },
            language_instruction=self.language_instruction,
        )

    def step(self, action: dict[str, SingleAct]) -> tuple[Obs, float, bool, bool, dict]:
        # change gripper to libero format (-1, 1) where -1 is open
        assert len(action) == 1, "Libero expects a single robot action"
        _, robot_action = next(iter(action.items()))
        if robot_action.gripper is None:
            raise ValueError("Libero expects a gripper value in SingleAct.gripper")
        act = np.concatenate(
            [
                np.asarray(robot_action.action, dtype=np.float32),
                np.asarray([robot_action.gripper], dtype=np.float32),
            ]
        )
        act[-1] = (1 - act[-1]) * 2 - 1.0
        obs, reward, done, info = self.env.step(act)
        success = self.env.check_success()
        return self.translate_obs(obs), reward, success, done, info

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Obs, dict[str, Any]]:
        if seed is not None:
            self.env.seed(seed)
        obs = self.env.reset()
        init_states = self.task_suite.get_task_init_states(
            self.task_id
        )  # for benchmarking purpose, we fix the a set of initial states
        init_state_id = 0
        self.env.set_init_state(init_states[init_state_id])

        for robot in self.env.robots:
            robot.controller.use_delta = True
        for _ in range(self.reset_steps):
            # steps the environment to filter out falling objects
            obs, _, _, _ = self.env.step(
                np.zeros(8) if "JOINT" in self.env_kwargs.get("controller", "OSC_POSE") else np.zeros(7)
            )

        if self.control_mode == "absolute":
            for robot in self.env.robots:
                robot.controller.use_delta = False
        elif self.control_mode == "relative":
            for robot in self.env.robots:
                robot.controller.use_delta = True
        else:
            raise ValueError(f"Invalid control mode: {self.control_mode}, use 'absolute' or 'relative'.")

        return self.translate_obs(obs), {}

    @property
    def language_instruction(self) -> str:
        return self._language_instruction


register_env("libero_10", Libero)
register_env("libero_90", Libero)
register_env("libero_100", Libero)
register_env("libero_spatial", Libero)
register_env("libero_object", Libero)
register_env("libero_goal", Libero)
