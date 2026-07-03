# VLAgents
VLAgents is a python library that allows to separate next action prediction from policy networks from action execution in simulated or real environments.
It defines an interface for policies and for environments.
The policies run independent in their own virtual environment, potentially on a different computer, and can be queried for an action (in principle similar to the chatgpt api).

Why is this useful?
- Separation of dependencies by using two different python environments: Some times dependencies contradict e.g. pytorch and jax
- Some robot hardware requires a real time linux kernel which does not easily allow you to use an Nvidia GPU.
- Separate deployment and model code

This library is a byproduct of the [Refined Policy Distillation (RPD)](https://refined-policy-distillation.github.io/) paper which distilled VLAs into expert policies using Reinforcement Learning.
The work also includes a section on related engineering challenges regarding jax and pytorch.

## Installation

### Pip Installation (Recommended)
```shell
pip install vlagents
```

### Local Installation
```shell
git clone https://https://github.com/RobotControlStack/vlagents.git
cd vlagents
pip install -ve .
```


### Environment and Policy Installation
On top of vlagents you can then install a simulation environment where the agent acts.
We currently the following environments:
- [maniskill](https://github.com/haosulab/ManiSkill)
- [robot control stack](https://github.com/RobotControlStack/robot-control-stack)
- [duobench](https://github.com/RobotControlStack/duobench)
- [libero](https://github.com/Lifelong-Robot-Learning/LIBERO)


In order to avoid dependency conflicts, use a second conda/pip environment to install your policy.
We currently support the following policies:
- [octo](https://github.com/octo-models/octo)
- [openvla](https://github.com/openvla/openvla)
- [openpi](https://github.com/Physical-Intelligence/openpi)
- [vjepa2-ac](https://github.com/facebookresearch/vjepa2)
- [diffusion policy](https://github.com/real-stanford/diffusion_policy)
- [lerobot policies](https://github.com/huggingface/lerobot/tree/main/src/lerobot/policies)


### LeRobot
```shell
pip install 'lerobot[all]'
```



### Octo
To use Octo as an agent/policy you need to create a new conda environment:
```shell
conda create -n octo python=3.10
conda activate octo
conda install nvidia/label/cuda-11.8.0::cuda --no-channel-priority
conda install conda-forge::cudnn=8.9
# octo dependencies
pip install git+https://github.com/octo-models/octo.git@241fb3514b7c40957a86d869fecb7c7fc353f540
pip install -r vlagents/utils/fixed_octo_requirements.txt
# for gpu support:
pip install --upgrade "jax[cuda11_pip]==0.4.20" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

Verify that the jax installation was successful and that jax finds your gpu.
Open a python shell in the same conda env and type
```python
from jax.lib import xla_bridge
# this should output "gpu" if the gpu installation was successful
print(xla_bridge.get_backend().platform)
```


Install the vlagents library on top:
```shell
pip install git+https://github.com/juelg/vlagents.git
```

For more details, see the [Octo github page](https://github.com/octo-models/octo).

#### Troubleshooting
If pip complains about dependency issues than it might have happened that torch somehow slipped in.
Check if you have any torch packages installed by
```shell
pip freeze | grep torch
# if any, uninstall them e.g.
pip uninstall arm_pytorch_utilities
pip uninstall pytorch-seed
pip uninstall pytorch_kinematics
```

### OpenVLA
To use OpenVLA, create a new conda environment:
```shell
conda create -n openvla python=3.10 -y
conda activate openvla
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y
```

Install [flash attention](https://github.com/Dao-AILab/flash-attention):
```shell
pip install packaging ninja
ninja --version; echo $?  # Verify Ninja --> should return exit code "0"
pip install "flash-attn==2.5.5" --no-build-isolation
# if you run into issues try `pip cache remove flash_attn` first
```

Install OpenVLA
```shell
pip install git+https://github.com/openvla/openvla.git@46b752f477cc5773cc1234b2e82c0e2130e4e890
```

Install the vlagents library on top:
```shell
pip install git+https://github.com/juelg/vlagents.git
```

For more details, see the [OpenVLA github page](https://github.com/openvla/openvla).

### OpenPi / Pi0
To use OpenPi, create a new conda environment:
```shell
conda create -n openpi python=3.11 -y
conda activate openpi
```
Clone the repo and install it.
```shell
git clone --recurse-submodules git@github.com:Physical-Intelligence/openpi.git
# Or if you already cloned the repo:
git submodule update --init --recursive
# install dependencies
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```
For more details see [openpi's github](https://github.com/Physical-Intelligence/openpi).

### vjEPA2-ac
To use VJEPA2-AC, create a new conda environment:
```shell
conda create -n vjepa2 python=3.12
conda activate vjepa2
```
Clone the repo and install it.
```shell
git clone git@github.com:facebookresearch/vjepa2.git
cd vjepa2
pip install -e .

pip install git+https://github.com/juelg/vlagents.git
pip install -ve .

```

### Diffusion Policy
Currently located on the branch `diffusion_policy`.

## Usage
To start an vlagents server use the `start-server` command where `kwargs` is a dictionary of the constructor arguments of the policy you want to start e.g.
```shell
# lerobot act (n_action_steps is the executed horizon of the action chunk)
python -m vlagents start-server lerobot --port 8080 --host 0.0.0.0 --kwargs '{"policy_name": "act", "checkpoint_path": "<path to pretrained_model>", "n_action_steps": 1}'

# lerobot pi05
python -m vlagents start-server lerobot --port 20000 --host 0.0.0.0 --kwargs '{"policy_name": "pi05", "checkpoint_path": "<path to pretrained_model>", "n_action_steps": 1}'

# lerobot xvla
uv run python -m vlagents start-server lerobot --port 20000 --host 0.0.0.0 --kwargs '{"policy_name": "xvla", "checkpoint_path": "<path to pretrained_model>", "n_action_steps": 1, "rename_map": {"head": "image", "left_wrist": "image2", "right_wrist": "image3"}}'

# maniflow / hvla
python -m vlagents start-server maniflow --port 8080 --host 0.0.0.0 --kwargs '{"checkpoint_path": "<path to model.safetensors>", "device": "cuda:0", "rename_map": {"base": "head_rgb"}}'

# octo
python -m vlagents start-server octo --host localhost --port 8080 --kwargs '{"checkpoint_path": "hf://Juelg/octo-base-1.5-finetuned-maniskill", "checkpoint_step": None, "horizon": 1, "unnorm_key": []}'

# openvla
python -m vlagents start-server openvla --host localhost --port 8080 --kwargs '{"checkpoint_path": "Juelg/openvla-7b-finetuned-maniskill", "device": "cuda:0", "attn_implementation": "flash_attention_2", "unnorm_key": "maniskill_human:7.0.0", "checkpoint_step": 40000}'

# openpi
python -m vlagents start-server openpi --port=8080 --host=localhost --kwargs='{"checkpoint_path": "<path to checkpoint>/{checkpoint_step}", "model_name": "pi0_rcs", "checkpoint_step": <checkpoint_step>}' # leave "{checkpoint_step}" it will be replaced, "model_name" is the key for the training config

# vjepa2-ac
python -m vlagents start-server vjepa --port=20997 --host=0.0.0.0 --kwargs='{"cfg_path": "configs/inference/vjepa2-ac-vitg/<your_config>.yaml", "model_name": "vjepa2_ac_vit_giant", "default_checkpoint_path": "../.cache/torch/hub/checkpoints/vjepa2-ac-vitg.pt"}'
```


There is also the `run-eval-during-training` command to evaluate a model during training, so a single checkpoint.
The `run-eval-post-training` command evaluates a range of checkpoints in parallel.
In both cases environment and arguments as well as policy and arguments and wandb config for logging can be passed as CLI arguments.


## Adding your own environment
```python
from vlagents.evaluator_envs import EvaluatorEnv, Obs, Act
from typing import Any

class YourEnv(EvaluatorEnv):

    def translate_obs(self, obs: dict[str, Any]) -> Obs:
        # translated your observation
        return Obs()

    def step(self, action: Act) -> tuple[Obs, float, bool, bool, dict]:
        # step your env
        obs, reward, success, truncated, info = self.env.step(action)
        return self.translate_obs(obs), reward, success, truncated, info

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Obs, dict[str, Any]]:
        obs, info = self.env.reset()
        return self.translate_obs(obs), info

    @property
    def language_instruction(self) -> str:
        # return task instruction
        return "pick up the cube"

    @staticmethod
    def do_import():
        # do imports required by your env
        import libero

EvaluatorEnv.register("your-env-id", YourEnv)
```

## Adding your own policy
```python
from vlagents.policies import Agent, AGENTS
from vlagents.evaluator_envs import Obs, Act
from typing import Any
import numpy as np

class YourAgent(Agent):
    def initialize(self):
        # heavy initialization, e.g. loading models
        pass

    def act(self, obs: Obs) -> Act:
        # your forward pass
        return Act(action=np.zeros(7, dtype=np.float32), done=False, info={})

    def reset(self, obs: Obs, instruction: Any, **kwargs) -> dict[str, Any]:
        # reset model if it has state and return info dict
        return {}

    def close(self, *args, **kwargs):
        pass
AGENTS["your-agent-id"] = YourAgent
```



## Contribution

### New Policy
In order to extend the library with a new policy network, extend the `Agent` class in [policies.py](src/vlagents/policies.py).
It is important to only invoke policy specific imports in the class functions, as each policy can have its own dependencies.


### New Environment
In order to extend the library with a new agent environment, extend the `EvaluatorEnv` class in [evaluator_envs.py](src/vlagents/evaluator_envs.py).


### Developer Tools
Install the following dev dependencies:
```shell
pip install 'pip>=25.1'
pip install --group dev
```

The following dev tools are provided:
```shell
# format the code
make format

# lint the code
make lint

# run tests
make test
```

## Citation
If you find the agent useful for your work, please consider citing the original works behind it:
```
@inproceedings{juelg2025refinedpolicydistillationvla,
    title={{Refined Policy Distillation}: {F}rom {VLA} Generalists to {RL} Experts}, 
    author={Tobias J{\"u}lg and Wolfram Burgard and Florian Walter},
    year={2025},
    booktitle={Proc.~of the IEEE/RSJ Int.~Conf.~on Intelligent Robots and Systems (IROS)}
}
@misc{juelg2026vlagentspolicyserverefficient,
      title={VLAgents: A Policy Server for Efficient VLA Inference}, 
      author={Tobias J{\"u}lg and Khaled Gamal and Nisarga Nilavadi and Pierre Krack and Seongjin Bien and Michael Krawez and Florian Walter and Wolfram Burgard},
      year={2026},
      howpublished={\url{https://arxiv.org/abs/2601.11250}}
}
```