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
from multiprocessing import Pool
from pathlib import Path
from time import sleep
from typing import Any

import gymnasium as gym
import numpy as np
from PIL import Image
from simple_slurm import Slurm
from tqdm import tqdm

from vlagents.client import RemoteAgent
from vlagents.policies import Act, Agent, Obs
from vlagents.wrappers import HumanCameraWrapper

logging.basicConfig(
    format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)


class EvaluatorEnv(ABC):
    ENVS: dict[str, "EvaluatorEnv"] = {}

    def __init__(self, env_id: str, **env_kwargs) -> None:
        self.do_import()
        self.env = gym.make(env_id, **env_kwargs)
        self.env_id = env_id

    def step(self, action: Act) -> tuple[Obs, float, bool, bool, dict]:
        raise NotImplementedError

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Obs, dict[str, Any]]:
        raise NotImplementedError

    @property
    def language_instruction(self) -> str:
        raise NotImplementedError

    @staticmethod
    def register(env_id: str, env: "EvaluatorEnv") -> None:
        EvaluatorEnv.ENVS[env_id] = env

    @staticmethod
    def make(env_id: str, **env_kwargs) -> "EvaluatorEnv":
        return EvaluatorEnv.ENVS[env_id](env_id, **env_kwargs)

    @staticmethod
    def do_import():
        raise NotImplementedError


class RCSDuoBench(EvaluatorEnv):
    INSTRUCTIONS = {}

    def __init__(self, env_id, **env_kwargs):
        self.robot_keys: str = env_kwargs.pop("robot_keys", ["left", "right"])
        self.control_mode: str = env_kwargs.pop("control_mode", "joints")
        self._instruction: str | None = None
        super().__init__(env_id, **env_kwargs)

    def translate_obs(self, obs: dict[str, Any]) -> Obs:
        cameras = {}
        for key in obs["frames"]:
            cameras[key] = obs["frames"][key]["rgb"]["data"]
            cameras[key] = np.array(Image.fromarray(cameras[key]).resize((224, 224), Image.Resampling.BILINEAR))
        state = []
        for key in self.robot_keys:
            state.append(obs[key]["joints"])
            state.append(obs[key]["gripper"])

        return Obs(
            cameras=cameras,
            gripper=None,
            state=np.concatenate(state),
            info={"high_res_cameras": {key: obs["frames"][key]["rgb"]["data"] for key in obs["frames"]}},
        )

    def step(self, action: Act) -> tuple[Obs, float, bool, bool, dict]:
        assert (
            len(action.action.shape) == 1
        ), "this function cannot deal with batches or action chunks, please return single actions"
        env_action = {}
        for idx, robot in enumerate(self.robot_keys):
            if self.control_mode == "joints":
                env_action[robot] = {
                    "joints": action.action[idx * 8 : idx * 8 + 7],
                    "gripper": action.action[idx * 8 + 7 : idx * 8 + 8],
                }
            else:
                env_action[robot] = {
                    "xyzrpy": action.action[idx * 7 : idx * 7 + 6],
                    "gripper": action.action[idx * 7 + 6 : idx * 7 + 7],
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


class ManiSkill(EvaluatorEnv):
    INSTRUCTIONS = {
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
            cameras=dict(rgb_side=obs["sensor_data"]["base_camera"]["rgb"].squeeze(0).numpy()),
            # gripper=float(not obs["extra"]["is_grasped"]),
        )

    def step(self, action: Act) -> tuple[Obs, float, bool, bool, dict]:
        # includes horizon
        # careful with gripper action: the model needs to be trained on [-1, 1] interval

        a = copy.copy(action.action[0])
        # a[-1] = -1.0 if a[-1] < 0.9 else 1.0
        if self.env_id == "PushT-v1":
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


EvaluatorEnv.register("LiftPegUpright-v1", ManiSkill)
EvaluatorEnv.register("PegInsertionSide-v1", ManiSkill)
EvaluatorEnv.register("PickCube-v1", ManiSkill)
EvaluatorEnv.register("PlugCharger-v1", ManiSkill)
EvaluatorEnv.register("PullCube-v1", ManiSkill)
EvaluatorEnv.register("PullCubeTool-v1", ManiSkill)
EvaluatorEnv.register("PushCube-v1", ManiSkill)
EvaluatorEnv.register("PushT-v1", ManiSkill)
EvaluatorEnv.register("RollBall-v1", ManiSkill)
EvaluatorEnv.register("StackCube-v1", ManiSkill)
EvaluatorEnv.register("PokeCube-v1", ManiSkill)


class Libero(EvaluatorEnv):

    def __init__(self, env_id: str, reset_steps: int = 14, **env_kwargs) -> None:
        """
        For supported env_kwargs checkout ControlEnv class in libero.
        We add the following env_kwargs on top:
        - task_id (int): libero task id for given task suite. The number of tasks per task suite can checked with Libero.n_tasks(env_id). Defaults to 0.
        - control_mode (str): either 'relative' or 'absolute'. Defaults to 'relative'.

        """
        logging.info("Creating Libero env")
        self.env_kwargs = env_kwargs
        self.reset_steps = reset_steps
        self.control_mode = self.env_kwargs.pop("control_mode", "relative")
        self.env, self._language_instruction, self.task_name, self.task_suite, self.task_id, self.task = self._make_gym(
            env_id, **self.env_kwargs
        )
        logging.info(
            f"Created Libero env, task suite: {env_id}, task id: {self.task_id}, task name {self.task_name}, instruction: {self._language_instruction}"
        )
        self.env_id = env_id

    @staticmethod
    def n_tasks(env_id: str) -> int:
        from libero.libero import benchmark, get_libero_path

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

    def translate_obs(self, obs: dict[str, Any]) -> Obs:
        return Obs(
            cameras=dict(rgb_side=obs["agentview_image"][::-1], rgb_wrist=obs["robot0_eye_in_hand_image"][::-1]),
            gripper=obs["robot0_gripper_qpos"] / 0.04,  # normalize
        )

    def step(self, action: Act) -> tuple[Obs, float, bool, bool, dict]:
        # change gripper to libero format (-1, 1) where -1 is open
        act = np.copy(action.action)
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


EvaluatorEnv.register("libero_10", Libero)
EvaluatorEnv.register("libero_90", Libero)
EvaluatorEnv.register("libero_100", Libero)
EvaluatorEnv.register("libero_spatial", Libero)
EvaluatorEnv.register("libero_object", Libero)
EvaluatorEnv.register("libero_goal", Libero)


@dataclass
class EvalConfig:
    env_id: str
    env_kwargs: dict[str, Any]
    max_steps_per_episode: int = 100
    seed: int = 42
    same_machine: bool = False
    jpeg_encoding: bool = False


@dataclass
class AgentConfig:
    host: str
    agent_name: str
    agent_kwargs: dict[str, Any]
    python_path: str = "python"
    """modify this if you want to use a specific python environment """
    port: int = 8080


def _write_camera_mp4(frames: list[np.ndarray], output_path: Path, fps: int = 30) -> None:
    if not frames:
        return

    height, width = frames[0].shape[:2]
    process = subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(fps),
            "-i",
            "-",
            "-an",
            "-vf",
            "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output_path),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    assert process.stdin is not None
    for frame in frames:
        process.stdin.write(np.ascontiguousarray(frame).astype(np.uint8).tobytes())
    process.stdin.close()
    process.wait()


def single_eval(
    env: EvaluatorEnv, agent: Agent, max_steps: int, ith_episode: int, start_seed: int
) -> tuple[list[float], list[float], list[float]]:
    logging.debug(f"Starting evaluation")
    obs, _ = env.reset(seed=start_seed + ith_episode)  # ensure different seed for each episode
    cameras = obs.info.pop("high_res_cameras", obs.cameras)
    logging.debug(f"Reset env")
    agent.reset(copy.deepcopy(obs), env.language_instruction)
    logging.debug(f"Reset agent")
    done = False
    truncated = False
    step = 0.0
    rewards = []
    im = []
    while not done and not truncated and max_steps > step:
        action = agent.act(obs)
        obs, reward, done, truncated, _ = env.step(action)
        cameras = obs.info.pop("high_res_cameras", obs.cameras)
        reward = float(reward)
        done, truncated = bool(done), bool(truncated)
        step += 1
        rewards.append(reward)
        im.append(cameras)

    cam_path = os.environ.get("CAM_PATH", None)
    if cam_path is not None and im:
        output_dir = Path(os.environ["CAM_PATH"]) / env.env_id
        output_dir.mkdir(exist_ok=True, parents=True)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        for camera in im[0].keys():
            _write_camera_mp4(
                [img[camera] for img in im],
                output_dir / f"{ith_episode}_{camera}_{timestamp}.mp4",
            )

    env.reset()
    logging.debug(f"Finished evaluation with {step} steps and reward {reward}, success {done}")
    # success, last reward and number of steps
    return done, rewards, step


per_process_cache = {}


def create_env_agent(agent_config: AgentConfig, cfg: EvalConfig) -> tuple[EvaluatorEnv, RemoteAgent]:
    logging.debug(f"retrieving env {cfg.env_id} and agent")
    key = (cfg.env_id, agent_config.host, agent_config.port)
    if key not in per_process_cache:
        logging.info(f"env {cfg.env_id} not available, creating new env and agent")
        env = EvaluatorEnv.make(cfg.env_id, **cfg.env_kwargs)
        logging.info("done creating env")
        agent = RemoteAgent(
            agent_config.host,
            agent_config.port,
            agent_config.agent_name,
            on_same_machine=cfg.same_machine,
            jpeg_encoding=cfg.jpeg_encoding,
        )
        logging.info("done creating agent")
        per_process_cache[key] = (env, agent)
    return per_process_cache[key]


def run_episode(args: tuple[int, list[EvalConfig], int, AgentConfig]) -> tuple[list[float], list[float], list[float]]:
    i, cfgs, episodes, agent_cfg = args
    cfg = cfgs[i // episodes]
    env, agent = create_env_agent(agent_cfg, cfg)
    # busy wait for server to finish initialization
    while not agent.is_initialized():
        logging.info("Waiting for agent to initialize...")
        sleep(5)
    return single_eval(env, agent, cfg.max_steps_per_episode, i, start_seed=cfg.seed)


def multi_eval(
    agent_cfg: AgentConfig, cfgs: list[EvalConfig], episodes: int = 100
) -> tuple[np.ndarray, list[list[list[float]]]]:
    # return is [envs, episodes, 3(success, reward, steps)], [envs, episodes, rewards for all steps in the episode]
    logging.info(f"Starting evaluation with {len(cfgs)} environments and {episodes} episodes each")

    # np.random.seed(cfgs[0].seed)
    args = [(i, cfgs, episodes, agent_cfg) for i in range(len(cfgs) * episodes)]
    single_results = [run_episode(arg) for arg in tqdm(args)]

    single_results_last_reward = np.array([(i[0], i[1][-1], i[2]) for i in single_results])

    # this works because row-major order
    # per_env_results = single_results.reshape(len(cfgs), episodes, 3)
    per_env_results_last_reward = single_results_last_reward.reshape(len(cfgs), episodes, 3)
    per_env_results_rewards = [
        [i[1] for i in single_results[i : i + episodes]] for i in range(0, len(single_results), episodes)
    ]
    return per_env_results_last_reward, per_env_results_rewards


@contextmanager
def start_server(
    agent_name: str,
    kwargs: dict[str, Any],
    port: int = 8080,
    host: str = "localhost",
    python_path: str = sys.executable,
):
    """Start the agent server in a subprocess as a context manager.

    This ensures that the server is properly stopped when exiting the context and
    that all logs are printed to the console.

    Args:
        agent_name (str): Name of the agent to start.
        kwargs (dict[str, Any]): Additional keyword arguments for the agent.
        port (int): Port to start the server on. Defaults to 8080.
        host (str): Host to bind the server to. Defaults to "localhost".
        python_path (str): Path to the Python interpreter to use. If you use conda you can look up the path with `conda info --envs`.
            It can also be a format string that will be formatted with the agent_name, e.g. "conda run -n {agent_name} python".
            Defaults to the current interpreter (`sys.executable`).
    """
    cmd = [
        python_path.format(agent_name=agent_name),
        "-m",
        "vlagents",
        "start-server",
        f"{agent_name}",
        f"--port={port}",
        f"--host={host}",
        f"--kwargs={json.dumps(kwargs)}",
    ]
    logging.info("Server starting: %s", " ".join(cmd))
    env = os.environ.copy()
    source_root = str(Path(__file__).resolve().parents[1])
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = source_root if not existing_pythonpath else f"{source_root}:{existing_pythonpath}"
    p = subprocess.Popen(cmd, env=env)
    try:
        yield p
    finally:
        # Stop the server no matter how we exit the with-block (success or exception).
        try:
            p.send_signal(subprocess.signal.SIGINT)
            p.wait(timeout=5)
        except Exception:
            pass
        if p.poll() is None:
            p.terminate()
            try:
                p.wait(timeout=3)
            except subprocess.TimeoutExpired:
                p.kill()
        logging.info("Server stopped")


def evaluation(
    agent_cfg: AgentConfig,
    eval_cfgs: list[EvalConfig],
    episodes: int = 100,
):
    per_process_cache.clear()
    logging.info(f"Starting evaluation with {agent_cfg.agent_name} and {agent_cfg.agent_kwargs}")
    try:
        with start_server(
            agent_cfg.agent_name, agent_cfg.agent_kwargs, agent_cfg.port, agent_cfg.host, agent_cfg.python_path
        ):
            sleep(30)
            res = multi_eval(agent_cfg, eval_cfgs, episodes)
    except Exception:
        # Ensures you SEE the client's stack trace and any logged errors.
        logging.exception("Client failed")
        raise

    logging.info(f"Results (success, reward, steps) for all envs: {res[0].mean(axis=1)}")
    logging.info(
        f"Mean reward for all envs: {[np.mean([np.mean(ep_rewards) for ep_rewards in env_rewards]) for env_rewards in res[1]]}"
    )
    # print indices of successful episodes
    for idx, env in enumerate(res[0]):
        logging.info(f"Env {eval_cfgs[idx].env_id} successful episodes: {np.where(env[:, 0])[0]}")
    return res


def run_eval(
    agent_cfg: AgentConfig,
    eval_cfgs: list[EvalConfig],
    wandb_entity: str,
    wandb_project: str,
    wandb_note: str,
    wandb_name: str,
    checkpoint_steps: list[int],
    slurm: Slurm,
    output_path: str,
    wandb_group: str | None = None,
    episodes: int = 100,
    n_processes: int | None = None,
    n_gpus: int = 1,
    python_path: str = "python",
):
    eval_cmd = shlex.quote(
        shlex.join(
            [
                "-m",
                "vlagents",
                "run-eval",
                f"--agent-cfg={json.dumps(asdict(agent_cfg))}",
                f"--episodes={episodes}",
                f"--n-processes={n_processes}",
                f"--eval-cfgs={json.dumps([asdict(cfg) for cfg in eval_cfgs])}",
                f"--wandb-group={wandb_group.replace(':', '_') if wandb_group else ''}",
                f"--wandb-project={wandb_project}",
                f"--wandb-entity={wandb_entity}",
                f"--wandb-note={wandb_note}",
                f"--wandb-name={wandb_name}",
                f"--n-gpus={n_gpus}",
                f"--steps={json.dumps(checkpoint_steps)}",
                f"--output-path={output_path}",
            ]
        )
    )

    python_path += eval_cmd
    slurm.sbatch(python_path)


def write_results(
    results: np.ndarray,
    rewards: list[list[list[float]]],
    eval_cfgs: list[EvalConfig],
    agent_cfg: AgentConfig,
    out: str = "",
    grouped_eval_cfgs: list[list[EvalConfig]] | None = None,
) -> str:
    # first read json, if not exists write empty list
    path = os.path.join(out, f"results_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json")
    if not os.path.exists(path):
        with open(path, "w") as f:
            json.dump([], f)
    with open(path, "r") as f:
        prev_results = json.load(f)
    assert isinstance(prev_results, list)

    flatten_rewards = [[item for sublist in env_rewards for item in sublist] for env_rewards in rewards]
    mean_rewards = [np.mean(env_rewards) for env_rewards in flatten_rewards]
    grouped_eval_cfgs = grouped_eval_cfgs or [[cfg] for cfg in eval_cfgs]

    for idx, (cfg, cfg_group) in enumerate(zip(eval_cfgs, grouped_eval_cfgs, strict=True)):
        success_mean, reward_mean, steps_mean = results[idx].mean(axis=0, keepdims=False)
        success_max, reward_max, steps_max = results[idx].max(axis=0, keepdims=False)
        success_min, reward_min, steps_min = results[idx].min(axis=0, keepdims=False)
        sucess_std, reward_std, steps_std = results[idx].std(axis=0, keepdims=False)
        success_median, reward_median, steps_median = np.median(results[idx], axis=0, keepdims=False)
        result_entry = {
            "success": {
                "mean": success_mean,
                "max": success_max,
                "min": success_min,
                "std": sucess_std,
                "median": success_median,
                "values": results[idx, :, 0].tolist(),
            },
            "reward_last_step": {
                "mean": reward_mean,
                "max": reward_max,
                "min": reward_min,
                "std": reward_std,
                "median": reward_median,
                "values": results[idx, :, 1].tolist(),
            },
            "rewards": {
                "mean": mean_rewards[idx],
                "values": rewards[idx],
            },
            "steps": {
                "mean": steps_mean,
                "max": steps_max,
                "min": steps_min,
                "std": steps_std,
                "median": steps_median,
                "values": results[idx, :, 2].tolist(),
            },
            "episodes": results.shape[1],
            "timestamp": datetime.datetime.now().isoformat(),
            "env_cfg": asdict(cfg),
            "agent_cfg": asdict(agent_cfg),
        }
        if len(cfg_group) > 1:
            result_entry["merged_env_cfgs"] = [asdict(group_cfg) for group_cfg in cfg_group]
        prev_results.append(result_entry)

    with open(path, "w") as f:
        json.dump(prev_results, f, indent=2)
    return path
