import datetime
import json
import logging
import os
import shlex
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from time import sleep
from typing import Any

import numpy as np
from simple_slurm import Slurm
from tqdm import tqdm

from vlagents.client import RemoteAgent
from vlagents.envs.interface import AgentConfig, EvalConfig, EvalEnv
from vlagents.policies.interface import Agent

logging.basicConfig(
    format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)


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
    env: EvalEnv, agent: Agent, max_steps: int, ith_episode: int, start_seed: int
) -> tuple[list[float], list[float], list[float]]:
    logging.debug(f"Starting evaluation")
    obs, _ = env.reset(seed=start_seed + ith_episode)  # ensure different seed for each episode
    if obs.language_instruction is None:
        obs.language_instruction = env.language_instruction
    single_obs = next(iter(obs.obs.values()))
    cameras = single_obs.info.pop("high_res_cameras", single_obs.cameras)
    logging.debug(f"Reset env")
    done = False
    truncated = False
    step = 0.0
    rewards = []
    im = []
    while not done and not truncated and max_steps > step:
        if obs.language_instruction is None:
            obs.language_instruction = env.language_instruction
        obs, reward, done, truncated, _ = env.chunk_step(agent.act(obs), max_steps=max_steps - int(step))
        if obs.language_instruction is None:
            obs.language_instruction = env.language_instruction
        single_obs = next(iter(obs.obs.values()))
        cameras = single_obs.info.pop("high_res_cameras", single_obs.cameras)
        reward = float(reward)
        done, truncated = bool(done), bool(truncated)
        step += env.last_chunk_steps
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


def create_env_agent(agent_config: AgentConfig, cfg: EvalConfig) -> tuple[EvalEnv, RemoteAgent]:
    logging.debug(f"retrieving env {cfg.env_id} and agent")
    key = (cfg.env_id, agent_config.host, agent_config.port)
    if key not in per_process_cache:
        logging.info(f"env {cfg.env_id} not available, creating new env and agent")
        env = EvalEnv.make(cfg.env_id, execution_horizon=cfg.execution_horizon, **cfg.env_kwargs)
        logging.info("done creating env")
        agent = RemoteAgent(
            agent_config.host,
            agent_config.port,
            agent_config.agent_name,
            on_same_machine=cfg.same_machine,
            jpeg_encoding=cfg.jpeg_encoding,
            image_size=cfg.image_size,
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
