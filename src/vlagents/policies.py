import base64
import json
import logging
import os
from collections import deque
from dataclasses import dataclass, field
from functools import partial, reduce
from multiprocessing import resource_tracker, shared_memory
from operator import getitem
from pathlib import Path
from typing import Any, Union

import numpy as np
import simplejpeg
from PIL import Image


@dataclass(kw_only=True)
class SharedMemoryPayload:
    shm_name: str
    shape: tuple[int, ...]
    dtype: str = "uint8"


class CameraDataType:
    SHARED_MEMORY = "shared_memory"
    JPEG_ENCODED = "jpeg_encoded"
    RAW = "raw"


@dataclass(kw_only=True)
class SingleObs:
    cameras: dict[str, np.ndarray | SharedMemoryPayload | str] = field(default_factory=dict)
    camera_data_type: str = CameraDataType.RAW
    gripper: float | None = None
    joints: np.ndarray | None = None
    # translation in m and rotation around x, y, z axes in radians
    xyzrpy: np.ndarray | None = None
    # translation in m and quaternion in (x, y, z, w) format
    tquat: np.ndarray | None = None
    info: dict[str, Any] = field(default_factory=dict)


@dataclass(kw_only=True)
class Obs:
    # dictionary for multiple robot arms
    obs: dict[str, SingleObs] = field(default_factory=dict)
    language_instruction: str | None = None
    goal_image: np.ndarray | SharedMemoryPayload | str | None = None
    goal_image_data_type: str = CameraDataType.RAW


@dataclass(kw_only=True)
class SingleAct:
    action: np.ndarray
    gripper: float | None = None
    done: bool = False
    info: dict[str, Any] = field(default_factory=dict)


@dataclass(kw_only=True)
class Act:
    # action chunk with dictionary for multiple robot arms
    acts: list[dict[str, SingleAct]] = field(default_factory=list)


class Agent:
    def __init__(
        self,
        default_checkpoint_path: str,
        checkpoint_path: str | None = None,
        checkpoint_step: int | None = None,
    ) -> None:
        self.checkpoint_step = checkpoint_step
        self.default_checkpoint_path = default_checkpoint_path
        self.checkpoint_path = checkpoint_path
        self.instruction: str | None = None
        self.step = -1
        self._shm: dict[str, shared_memory.SharedMemory] = {}

    def initialize(self):
        # heavy initialization, e.g. loading models
        pass

    def _decode_image_payload(
        self,
        payload: np.ndarray | SharedMemoryPayload | str,
        data_type: str,
    ) -> np.ndarray:
        if data_type == CameraDataType.RAW:
            assert isinstance(payload, np.ndarray)
            return payload
        if data_type == CameraDataType.SHARED_MEMORY:
            assert isinstance(payload, SharedMemoryPayload)
            if payload.shm_name not in self._shm:
                self._shm[payload.shm_name] = shared_memory.SharedMemory(payload.shm_name)
            shm = self._shm[payload.shm_name]
            return np.ndarray(payload.shape, dtype=payload.dtype, buffer=shm.buf)
        if data_type == CameraDataType.JPEG_ENCODED:
            assert isinstance(payload, str)
            return simplejpeg.decode_jpeg(base64.urlsafe_b64decode(payload))
        raise ValueError(f"Unsupported camera data type: {data_type}")

    def _to_numpy(self, obs: Obs) -> Obs:
        """Decode camera payloads in-place for every robot and goal image."""
        for single_obs in obs.obs.values():
            single_obs.cameras = {
                camera_name: self._decode_image_payload(camera_data, single_obs.camera_data_type)
                for camera_name, camera_data in single_obs.cameras.items()
            }
            single_obs.camera_data_type = CameraDataType.RAW

        if obs.goal_image is not None:
            obs.goal_image = self._decode_image_payload(obs.goal_image, obs.goal_image_data_type)
            obs.goal_image_data_type = CameraDataType.RAW
        return obs

    def _require_single_arm(self, obs: Obs) -> tuple[str, SingleObs]:
        if len(obs.obs) != 1:
            raise ValueError(f"{type(self).__name__} currently supports exactly one arm, got {list(obs.obs.keys())}")
        robot_name, single_obs = next(iter(obs.obs.items()))
        return robot_name, single_obs

    def _single_obs_state(self, single_obs: SingleObs, *, include_gripper: bool = True) -> np.ndarray:
        state_parts: list[np.ndarray] = []
        if single_obs.joints is not None:
            state_parts.append(np.asarray(single_obs.joints, dtype=np.float32))
        if include_gripper and single_obs.gripper is not None:
            state_parts.append(np.asarray([single_obs.gripper], dtype=np.float32))
        if not state_parts:
            raise ValueError(f"{type(self).__name__} requires joints and/or gripper in the observation")
        return np.concatenate(state_parts)

    def _single_step_act(
        self,
        robot_name: str,
        action: np.ndarray,
        *,
        gripper: float | None = None,
        done: bool = False,
        info: dict[str, Any] | None = None,
    ) -> Act:
        return Act(
            acts=[
                {
                    robot_name: SingleAct(
                        action=np.asarray(action, dtype=np.float32),
                        gripper=None if gripper is None else float(gripper),
                        done=done,
                        info={} if info is None else info,
                    )
                }
            ]
        )

    def _chunk_act(
        self,
        robot_name: str,
        action_chunk: np.ndarray,
        *,
        grippers: np.ndarray | list[float] | None = None,
        infos: list[dict[str, Any] | None] | None = None,
        done: bool = False,
    ) -> Act:
        actions = np.asarray(action_chunk, dtype=np.float32)
        if actions.ndim == 1:
            actions = actions[None, :]
        if grippers is None:
            gripper_values = [None] * len(actions)
        else:
            gripper_array = np.asarray(grippers, dtype=np.float32).reshape(-1)
            if len(gripper_array) != len(actions):
                raise ValueError("grippers must have the same length as the action chunk")
            gripper_values = [float(value) for value in gripper_array]
        info_values = infos or [None] * len(actions)
        if len(info_values) != len(actions):
            raise ValueError("infos must have the same length as the action chunk")
        return Act(
            acts=[
                {
                    robot_name: SingleAct(
                        action=actions[idx],
                        gripper=gripper_values[idx],
                        done=done and idx == len(actions) - 1,
                        info={} if info_values[idx] is None else info_values[idx],
                    )
                }
                for idx in range(len(actions))
            ]
        )

    def act(self, obs: Obs) -> Act:
        self.instruction = obs.language_instruction
        self.step += 1
        self._to_numpy(obs)
        return Act(acts=[])

    def __enter__(self):
        pass

    def __exit__(self, *args, **kwargs):
        self.close()

    def close(self, *args, **kwargs):
        for shm in self._shm.values():
            shm.close()
            resource_tracker.unregister(shm._name, "shared_memory")
        self._shm = {}


class TestAgent(Agent):
    def __init__(self, **kwargs) -> None:
        super().__init__(default_checkpoint_path="", **kwargs)
        self.i = 0

    def act(self, obs: Obs) -> Act:
        super().act(obs)
        assert len(obs.obs) == 1, "TestAgent currently expects a single robot observation"
        robot_name, robot_obs = next(iter(obs.obs.items()))
        info = {
            "shapes": {k: v.shape for k, v in robot_obs.cameras.items()},
            "dtype": {k: v.dtype.name for k, v in robot_obs.cameras.items()},
            "data": {k: v for k, v in robot_obs.cameras.items()},
        }
        a = Act(
            acts=[
                {
                    robot_name: SingleAct(
                        action=np.array([0, 0, 0, 0, 0, 0], dtype=np.float32),
                        gripper=float(self.i % 2),
                        done=False,
                        info=info,
                    )
                }
            ]
        )
        self.i += 1
        return a


class LeRobotPolicy(Agent):
    def __init__(
        self,
        policy_name: str = "pi05",
        default_checkpoint_path: str = "lerobot/pi05_base",
        device: str = "cuda:0",
        temporal_ensemble_coeff: float | None = None,
        rename_map: dict[str, str] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(default_checkpoint_path=default_checkpoint_path, **kwargs)

        self.policy_name = policy_name
        self.device = device
        self.temporal_ensemble_coeff = temporal_ensemble_coeff
        checkpoint_path = self.checkpoint_path or self.default_checkpoint_path
        if self.checkpoint_step is not None:
            checkpoint_path = checkpoint_path.format(checkpoint_step=self.checkpoint_step)
        self.path = checkpoint_path

        if rename_map is not None:
            self.rename_map = rename_map
        else:
            self.rename_map = {}

        # self.rename_map = {
        #     "head": "image",
        #     "left_wrist": "image2",
        #     "right_wrist": "image3",
        # }

    def initialize(self):
        from collections import deque

        import torch
        from lerobot.policies.factory import get_policy_class, make_pre_post_processors
        from torchvision.transforms import v2

        # from vlagents import train_xvla

        self.policy = get_policy_class(self.policy_name).from_pretrained(self.path)

        if self.policy_name == "act":
            from lerobot.policies.act.modeling_act import ACTTemporalEnsembler

            if self.temporal_ensemble_coeff is not None:
                self.policy.config.temporal_ensemble_coeff = self.temporal_ensemble_coeff
                self.policy.temporal_ensembler = ACTTemporalEnsembler(
                    self.temporal_ensemble_coeff,
                    self.policy.config.chunk_size,
                )
            elif hasattr(self.policy, "temporal_ensembler"):
                delattr(self.policy, "temporal_ensembler")

        self._expected_image_shapes = {
            key.removeprefix("observation.images."): tuple(feature.shape)
            for key, feature in self.policy.config.input_features.items()
            if key.startswith("observation.images.")
        }
        self._camera_transforms = {
            key: v2.Compose(
                [
                    v2.ToImage(),
                    v2.Resize((height, width)),
                    v2.ToDtype(torch.float32, scale=True),
                    v2.ToPureTensor(),
                ]
            )
            for key, (_, height, width) in self._expected_image_shapes.items()
        }
        # self.policy.config.device = self.device
        self.policy.to(self.device)
        self.policy.eval()

        preprocessor_overrides = {
            "device_processor": {"device": self.device},
            # "rename_observations_processor": {"rename_map": self.rename_map},
        }

        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=self.policy.config,
            pretrained_path=self.path,
            preprocessor_overrides=preprocessor_overrides,
        )

    def act(self, obs: Obs) -> Act:
        import torch

        super().act(obs)
        robot_name, single_obs = self._require_single_arm(obs)

        observation = {
            "observation.state": torch.as_tensor(np.array(self._single_obs_state(single_obs), copy=True)).to(
                torch.float32
            ),
            "task": obs.language_instruction,
        }

        for key, img_data in single_obs.cameras.items():
            renamed_key = self.rename_map.get(key, key)
            expected_shape = self._expected_image_shapes.get(renamed_key)
            assert expected_shape is not None
            observation[f"observation.images.{renamed_key}"] = self._camera_transforms[renamed_key](
                np.array(img_data, copy=True)
            )

        observation = self.preprocessor(observation)

        with torch.inference_mode():
            action = self.policy.predict_action_chunk(observation)
        action = self.postprocessor(action)

        if isinstance(action, torch.Tensor):
            action = action.detach().float().cpu().numpy()

        action_chunk = np.squeeze(action, axis=0)  # remove batch dimension
        if action_chunk.ndim == 1:
            action_chunk = action_chunk[None, :]
        if action_chunk.shape[-1] < 1:
            raise ValueError("LeRobot action chunk must include a gripper dimension")
        return self._chunk_act(robot_name, action_chunk[:, :-1], grippers=action_chunk[:, -1])


class VjepaAC(Agent):
    def __init__(
        self,
        cfg_path: str,
        model_name: str = "vjepa2_ac_vit_giant",
        default_checkpoint_path: str = "",
        **kwargs,
    ) -> None:
        super().__init__(default_checkpoint_path=default_checkpoint_path, **kwargs)
        import yaml

        self.cfg_path = cfg_path
        with open(self.cfg_path, "r") as f:
            self.cfg = yaml.safe_load(f)

        self.model_name = model_name

    def initialize(self):
        # torch import
        import torch

        # VJEPA imports
        from app.vjepa_droid.transforms import make_transforms
        from notebooks.utils.world_model_wrapper import WorldModel

        self.device = self.cfg.get("device", "cuda")
        self.goal_img = self.cfg.get("goal_img", "exp_1.png")

        # data config
        cfgs_data = self.cfg.get("data")
        crop_size = cfgs_data.get("crop_size", 256)

        # data augs
        cfgs_data_aug = self.cfg.get("data_aug")
        use_aa = cfgs_data_aug.get("auto_augment", False)
        horizontal_flip = cfgs_data_aug.get("horizontal_flip", False)
        motion_shift = cfgs_data_aug.get("motion_shift", False)
        ar_range = cfgs_data_aug.get("random_resize_aspect_ratio", [3 / 4, 4 / 3])
        rr_scale = cfgs_data_aug.get("random_resize_scale", [0.3, 1.0])
        reprob = cfgs_data_aug.get("reprob", 0.0)

        # cfgs_mpc_args config
        cfgs_mpc_args = self.cfg.get("mpc_args")
        self.rollout_horizon = cfgs_mpc_args.get("rollout_horizon", 2)
        samples = cfgs_mpc_args.get("samples", 25)
        topk = cfgs_mpc_args.get("topk", 10)
        cem_steps = cfgs_mpc_args.get("cem_steps", 1)
        momentum_mean = cfgs_mpc_args.get("momentum_mean", 0.15)
        momentum_mean_gripper = cfgs_mpc_args.get("momentum_mean_gripper", 0.15)
        momentum_std = cfgs_mpc_args.get("momentum_std", 0.75)
        momentum_std_gripper = cfgs_mpc_args.get("momentum_std_gripper", 0.15)
        maxnorm = cfgs_mpc_args.get("maxnorm", 0.075)
        verbose = cfgs_mpc_args.get("verbose", True)

        # Initialize transform (random-resize-crop augmentations)
        self.transform = make_transforms(
            random_horizontal_flip=horizontal_flip,
            random_resize_aspect_ratio=ar_range,
            random_resize_scale=rr_scale,
            reprob=reprob,
            auto_augment=use_aa,
            motion_shift=motion_shift,
            crop_size=crop_size,
        )

        # load model
        encoder, predictor = torch.hub.load(
            "./",
            self.model_name,
            source="local",
            pretrained=True,  # root of the vjepa source code  # model type
        )

        # load model to cuda
        encoder.to(self.device)
        predictor.to(self.device)

        # World model wrapper initialization
        tokens_per_frame = int((crop_size // encoder.patch_size) ** 2)
        self.world_model = WorldModel(
            encoder=encoder,
            predictor=predictor,
            tokens_per_frame=tokens_per_frame,
            mpc_args={
                "rollout": self.rollout_horizon,
                "samples": samples,
                "topk": topk,
                "cem_steps": cem_steps,
                "momentum_mean": momentum_mean,
                "momentum_mean_gripper": momentum_mean_gripper,
                "momentum_std": momentum_std,
                "momentum_std_gripper": momentum_std_gripper,
                "maxnorm": maxnorm,
                "verbose": verbose,
            },
            normalize_reps=True,
            device=self.device,
        )

        img = Image.open(self.goal_img)
        # [H, W, C] -> [T=1, H, W, C]
        goal_image = np.expand_dims(np.array(img), axis=0)
        # [T=1, H, W, C] -> [B=1, C, T, crop, patches]
        goal_image_tensor = torch.tensor(self.transform(goal_image)[None, :]).to(
            device=self.device, dtype=torch.float, non_blocking=True
        )
        with torch.no_grad():
            self.goal_rep = self.world_model.encode(goal_image_tensor)

    def act(self, obs: Obs) -> Act:
        # torch imports
        import torch

        super().act(obs)
        robot_name, single_obs = self._require_single_arm(obs)

        with torch.no_grad():
            side = single_obs.cameras["rgb_side"]
            # [H, W, C] -> [T=1, H, W, C]
            side = torch.permute(torch.as_tensor(side), (1, 2, 0)).unsqueeze(0)
            # [T=1, H, W, C] -> [B=1, C, T, crop, patches]
            input_image_tensor = (self.transform(side)[None, :]).to(
                device=self.device, dtype=torch.float, non_blocking=True
            )
            # encoder output shape: [B=1, num_tokens, dim]
            z_n = self.world_model.encode(input_image_tensor)

            if single_obs.xyzrpy is None:
                raise ValueError("VjepaAC requires xyzrpy in SingleObs.xyzrpy")
            xyzrpy = np.asarray(single_obs.xyzrpy, dtype=np.float32)
            gripper = float(single_obs.gripper if single_obs.gripper is not None else 0.0)
            # [xyzrpy(6), gripper(1)] -> [B=1, state_dim]
            s_n = (
                torch.tensor(np.concatenate((xyzrpy, [1 - gripper]), axis=0))
                .unsqueeze(0)
                .to(self.device, dtype=torch.float, non_blocking=True)
            )

            # predicted action chunk: [rollout_horizon, action_dim]
            actions = np.asarray(self.world_model.infer_next_action(z_n, s_n, self.goal_rep).cpu(), dtype=np.float32)
            # VJEPA uses the opposite gripper convention from vlagents.
            actions[:, -1] = 1 - actions[:, -1]

        return self._chunk_act(robot_name, actions[:, :-1], grippers=actions[:, -1])


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


class OpenVLAModel(Agent):
    # === Utilities ===
    SYSTEM_PROMPT = (
        "A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions."
    )

    def __init__(
        self,
        attn_implementation: str,
        device: str,
        unnorm_key: str,
        default_checkpoint_path: str = "openvla/openvla-7b",
        **kwargs,
    ) -> None:
        super().__init__(default_checkpoint_path=default_checkpoint_path, **kwargs)
        self.unnorm_key = unnorm_key
        logging.info(f"Using unnorm_key: {self.unnorm_key}")
        self.attn_implementation = attn_implementation
        if self.checkpoint_step is None or self.checkpoint_path is None:
            self.openvla_path = self.default_checkpoint_path
            logging.info(f"Using default checkpoint path: {self.openvla_path}")
            if self.unnorm_key != "viola":
                logging.warning(
                    "unnorm_key should be 'viola' when using default path, ignoring unnorm_key and setting it to 'viola'"
                )
                self.unnorm_key = "viola"
        else:
            self.openvla_path = self.checkpoint_path.format(checkpoint_step=self.checkpoint_step)
            logging.info(
                f"Using custom checkpoint path: {self.openvla_path} with checkpoint step: {self.checkpoint_step}"
            )
        self.device = device
        self.attn_implementation = attn_implementation

    def initialize(self):
        import torch
        from transformers import AutoModelForVision2Seq, AutoProcessor

        self.device = torch.device(self.device) if torch.cuda.is_available() else torch.device("cpu")

        # Load VLA Model using HF AutoClasses
        self.processor = AutoProcessor.from_pretrained(self.openvla_path, trust_remote_code=True)
        self.vla = AutoModelForVision2Seq.from_pretrained(
            self.openvla_path,
            attn_implementation=self.attn_implementation,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to(self.device)
        print("==========================")
        print(self.vla.norm_stats.keys())

        # [Hacky] Load Dataset Statistics from Disk (if passing a path to a fine-tuned model)
        if os.path.isdir(self.openvla_path):
            with open(Path(self.openvla_path) / "dataset_statistics.json", "r") as f:
                self.vla.norm_stats = json.load(f)

    def get_openvla_prompt(self, instruction: str, openvla_path: Union[str, Path]) -> str:
        if "v01" in openvla_path:
            return f"{self.SYSTEM_PROMPT} USER: What action should the robot take to {instruction.lower()}? ASSISTANT:"
        else:
            return f"In: What action should the robot take to {instruction.lower()}?\nOut:"

    def act(self, obs: Obs) -> Act:
        # no batch dimension here
        import torch

        super().act(obs)
        robot_name, single_obs = self._require_single_arm(obs)
        assert single_obs.cameras["rgb_side"].shape == (256, 256, 3), "wrong shape, use lanczos"
        image = single_obs.cameras["rgb_side"]
        unnorm_key = self.unnorm_key

        prompt = self.get_openvla_prompt(obs.language_instruction or "", self.openvla_path)
        inputs = self.processor(prompt, Image.fromarray(image).convert("RGB")).to(self.device, dtype=torch.bfloat16)
        action = np.asarray(self.vla.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False), dtype=np.float32)
        # OpenVLA returns a single step with gripper in the last dimension.
        return self._single_step_act(robot_name, action[:-1], gripper=float(action[-1]))


class OctoModel(Agent):
    """
    This model is trained with a window size of 2, predicting 7 dimensional actions 4 steps into the future.
    Observations and tasks conform to the following spec:

    Observations: {
        image_primary: ('batch', 'history_window', 256, 256, 3),
        image_wrist: ('batch', 'history_window', 128, 128, 3),
    }
    Tasks: {
        image_primary: ('batch', 256, 256, 3),
        image_wrist: ('batch', 128, 128, 3),
        language_instruction: {
            attention_mask: ('batch', 16),
            input_ids: ('batch', 16),
        },
    }

    At inference, you may pass in any subset of these observation and task keys, with a history window up to 2 timesteps.
    """

    def __init__(
        self,
        horizon: int = 2,
        unnorm_key: list[str] | None = None,
        default_checkpoint_path: str = "hf://rail-berkeley/octo-base-1.5",
        **kwargs,
    ) -> None:
        # default window size is 2 in octo
        # default unnorm is viola as it used the fr3
        super().__init__(default_checkpoint_path=default_checkpoint_path, **kwargs)
        self.horizon = horizon
        if unnorm_key is None:
            self.unnorm_key = []
        else:
            self.unnorm_key = unnorm_key

        # log checkpoint path and step and kwargs
        logging.info(f"checkpoint_path: {self.checkpoint_path}, checkpoint_step: {self.checkpoint_step}")
        logging.info(f"horizon: {self.horizon}")
        logging.info(f"unnorm_key: {self.unnorm_key}")
        logging.info(f"kwargs: {kwargs}")
        if self.checkpoint_path is None:
            self.octo_path = self.default_checkpoint_path
            logging.info(f"Using default checkpoint path: {self.octo_path}")
            if self.checkpoint_step is not None:
                logging.warning(
                    "checkpoint_step should be None when using default path, ignoring checkpoint_step and setting it to None"
                )
                self.checkpoint_step = None
            if self.unnorm_key != ["viola"]:
                logging.warning(
                    "unnorm_key should be ['viola'] when using default path, ignoring unnorm_key and setting it to ['viola']"
                )
                self.unnorm_key = ["viola"]
        else:
            self.octo_path = self.checkpoint_path
            logging.info(f"Using custom checkpoint path: {self.octo_path} with checkpoint step: {self.checkpoint_step}")
        logging.info(f"Using unnorm_key: {self.unnorm_key}")

    def initialize(self):
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = (
            "false"  # disable preallocation of memory in jax (might make it less efficient)
        )
        from octo.model.octo_model import OctoModel as _OctoModel
        from octo.utils.train_callbacks import supply_rng

        self.model = _OctoModel.load_pretrained(self.octo_path, self.checkpoint_step)

        window_size = self.model.example_batch["observation"]["timestep_pad_mask"].shape[1]
        if window_size < self.horizon:
            logging.warning(
                f"Horizon {self.horizon} is greater than the model's window size {window_size} with which the model has been trained with. "
            )

        self.trained_obs = self.model.example_batch["observation"].keys()

        logging.info("==========================")
        logging.info(self.model.dataset_statistics.keys())
        self.policy_fn = supply_rng(
            partial(
                self.model.sample_actions,
                unnormalization_statistics=reduce(getitem, self.unnorm_key, self.model.dataset_statistics)["action"],
            ),
        )

    def act(self, obs: Obs) -> Act:
        import jax
        from octo.utils.gym_wrappers import stack_and_pad

        super().act(obs)
        robot_name, single_obs = self._require_single_arm(obs)
        assert single_obs.cameras["rgb_side"].shape == (256, 256, 3), "wrong shape, use lanczos"
        image_obs = {"image_primary": single_obs.cameras["rgb_side"]}
        history = deque([image_obs] * self.horizon, maxlen=self.horizon)
        full_obs = stack_and_pad(history, self.horizon)
        task = self.model.create_tasks(texts=[obs.language_instruction or ""])

        actions = self.policy_fn(
            jax.tree_map(
                lambda x: x[None],
                full_obs,
            ),
            task,
        )
        action_chunk = np.asarray(actions[0, :, :], dtype=np.float32)
        return self._chunk_act(robot_name, action_chunk[:, :-1], grippers=action_chunk[:, -1])


class OctoActionDistribution(OctoModel):
    """
    this model does not support history window
    dont use self.step and self.episode as this model is not used sequentially
    """

    def __init__(self, **kwargs) -> None:
        assert kwargs["horizon"] == 1, "horizon must be 1 for OctoActionDistribution"
        super().__init__(**kwargs)

    def act(self, obs: Obs) -> Act:
        import jax.numpy as jnp

        Agent.act(self, obs)
        robot_name, single_obs = self._require_single_arm(obs)

        batch_size = single_obs.cameras["rgb_side"].shape[0]
        assert single_obs.cameras["rgb_side"].shape == (batch_size, 256, 256, 3), "wrong shape"
        num_samples = single_obs.info.get("num_samples", 1)

        x = jnp.array(single_obs.cameras["rgb_side"])
        x_expanded = jnp.expand_dims(x, 1)
        x_tiled = jnp.tile(x_expanded, (1, num_samples, 1, 1, 1))
        x_duplicated = x_tiled.reshape(-1, x.shape[1], x.shape[2], x.shape[3])
        full_obs = {
            "image_primary": jnp.expand_dims(x_duplicated, 1),
            "timestep_pad_mask": np.ones((batch_size * num_samples, 1)),
        }
        tasks = self.model.create_tasks(texts=[obs.language_instruction or ""] * batch_size * num_samples)
        actions = self.policy_fn(full_obs, tasks)
        actions = np.asarray(actions[:, 0, :].reshape(batch_size, num_samples, -1), dtype=np.float32)
        stds = np.std(actions, axis=1).astype(np.float32)
        means = np.mean(actions, axis=1).astype(np.float32)

        return Act(
            acts=[
                {
                    robot_name: SingleAct(
                        action=np.empty((0,), dtype=np.float32),
                        gripper=None,
                        done=False,
                        info={"means": means, "stds": stds, "actions": actions},
                    )
                }
            ]
        )


class OpenVLADistribution(OpenVLAModel):
    def act(self, obs: Obs) -> Act:
        import time

        import torch

        Agent.act(self, obs)
        robot_name, single_obs = self._require_single_arm(obs)
        batch_size = single_obs.cameras["rgb_side"].shape[0]

        images = single_obs.cameras["rgb_side"]
        actions = []
        unnorm_key = self.unnorm_key
        num_samples = single_obs.info.get("num_samples", 1)

        t1 = time.time()
        prompt = self.get_openvla_prompt(obs.language_instruction or "", self.openvla_path)

        x_expanded = np.expand_dims(images, 1)
        x_tiled = np.tile(x_expanded, (1, num_samples, 1, 1, 1))
        x_duplicated = x_tiled.reshape(-1, images.shape[1], images.shape[2], images.shape[3])

        for image in x_duplicated:
            inputs = self.processor(prompt, Image.fromarray(image).convert("RGB")).to(self.device, dtype=torch.bfloat16)
            actions.append(self.vla.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False))

        t2 = time.time()
        logging.info(f"needed time for {len(actions)} was {t2 - t1}s")

        actions = np.stack(actions).astype(np.float32).reshape(batch_size, num_samples, -1)
        means = np.mean(actions, axis=1).astype(np.float32)
        stds = np.std(actions, axis=1).astype(np.float32)
        return Act(
            acts=[
                {
                    robot_name: SingleAct(
                        action=np.empty((0,), dtype=np.float32),
                        gripper=None,
                        done=False,
                        info={"means": means, "stds": stds, "actions": actions},
                    )
                }
            ]
        )


AGENTS = dict(
    test=TestAgent,
    octo=OctoModel,
    lerobot=LeRobotPolicy,
    openvla=OpenVLAModel,
    octodist=OctoActionDistribution,
    openvladist=OpenVLADistribution,
    openpi=OpenPiModel,
    vjepa=VjepaAC,
)
