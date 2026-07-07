import base64
import copy
import json
import logging
import os
import sys
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
class Obs:
    cameras: dict[str, np.ndarray | SharedMemoryPayload | str] = field(default_factory=dict)
    camera_data_type: str = CameraDataType.RAW
    gripper: float | None = None
    # TODO: add context about what the state means, and its dimensions
    # theoratically it would be joints, xyzrpy and absolute or relative
    state: np.ndarray | None = None
    info: dict[str, Any] = field(default_factory=dict)


@dataclass(kw_only=True)
class Act:
    action: np.ndarray
    done: bool = False
    info: dict[str, Any] = field(default_factory=dict)


class Agent:
    def __init__(
        self, default_checkpoint_path: str, checkpoint_path: str | None = None, checkpoint_step: int | None = None
    ) -> None:
        self.instruction = None
        self.step = -1
        self.episode = -1
        self.checkpoint_step = checkpoint_step
        self.default_checkpoint_path = default_checkpoint_path
        self.checkpoint_path = checkpoint_path
        self._shm: dict[str, shared_memory.SharedMemory] = {}

    def initialize(self):
        # heavy initialization, e.g. loading models
        pass

    def _to_numpy(self, obs: Obs) -> Obs:
        """transparently uses shared memory if configured and modifies obs in place"""
        if obs.camera_data_type == CameraDataType.SHARED_MEMORY:
            camera_dict = {}
            for camera_name, camera_data in obs.cameras.items():
                assert isinstance(camera_data, SharedMemoryPayload)
                if camera_data.shm_name not in self._shm:
                    self._shm[camera_data.shm_name] = shared_memory.SharedMemory(camera_data.shm_name)
                camera_dict[camera_name] = np.ndarray(
                    camera_data.shape, dtype=camera_data.dtype, buffer=self._shm[camera_data.shm_name].buf
                )
            obs.cameras = camera_dict
        elif obs.camera_data_type == CameraDataType.JPEG_ENCODED:
            camera_dict = {}
            for camera_name, camera_data in obs.cameras.items():
                assert isinstance(camera_data, str)
                camera_dict[camera_name] = simplejpeg.decode_jpeg(base64.urlsafe_b64decode(camera_data))
            obs.cameras = camera_dict
        obs.camera_data_type = CameraDataType.RAW
        return obs

    def act(self, obs: Obs) -> Act:
        assert self.instruction is not None, "forgot reset?"
        self.step += 1
        self._to_numpy(obs)

        return Act(action=np.zeros(7, dtype=np.float32), done=False, info={})

    def reset(self, obs: Obs, instruction: Any, **kwargs) -> dict[str, Any]:
        logging.info(f"Resetting agent, new instruction: {instruction} ###############")
        self.step = 0
        self.episode += 1
        self.instruction = instruction
        self._to_numpy(obs)
        # info
        return {}

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
        # echo data back for testing
        info = {
            "shapes": {k: v.shape for k, v in obs.cameras.items()},
            "dtype": {k: v.dtype.name for k, v in obs.cameras.items()},
            "data": {k: v for k, v in obs.cameras.items()},
        }
        a = Act(action=np.array([0, 0, 0, 0, 0, 0, self.i % 2], dtype=np.float32), done=False, info=info)
        self.i += 1
        return a

    def reset(self, obs: Obs, instruction: Any, **kwargs) -> dict[str, Any]:
        super().reset(obs, instruction, **kwargs)
        info = {
            "shapes": {k: v.shape for k, v in obs.cameras.items()},
            "dtype": {k: v.dtype.name for k, v in obs.cameras.items()},
            "data": {k: v for k, v in obs.cameras.items()},
            "instruction": instruction,
        }
        return info


class LeRobotPolicy(Agent):

    def __init__(
        self,
        policy_name: str = "pi05",
        default_checkpoint_path: str = "lerobot/pi05_base",
        device: str = "cuda:0",
        n_action_steps: int = 30,
        temporal_ensemble_coeff: float | None = None,
        rename_map: dict[str, str] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(default_checkpoint_path=default_checkpoint_path, **kwargs)

        self.policy_name = policy_name
        self.device = device
        self.n_action_steps = n_action_steps
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
        self.policy.config.n_action_steps = self.n_action_steps

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

            if self.policy.config.temporal_ensemble_coeff is None:
                self.policy._action_queue = deque([], maxlen=self.policy.config.n_action_steps)

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

        observation = {
            "observation.state": torch.as_tensor(np.array(obs.state, copy=True)).to(torch.float32),
            "task": self.instruction,
        }

        for key, img_data in obs.cameras.items():
            expected_shape = self._expected_image_shapes.get(self.rename_map.get(key, key))
            assert expected_shape is not None
            observation[f"observation.images.{self.rename_map.get(key, key)}"] = self._camera_transforms[
                self.rename_map.get(key, key)
            ](np.array(img_data, copy=True))

        observation = self.preprocessor(observation)

        with torch.inference_mode():
            action = self.policy.select_action(observation)
            # action = self.policy.predict_action_chunk(observation)
        action = self.postprocessor(action)

        if isinstance(action, torch.Tensor):
            action = action.detach().float().cpu().numpy()

        action = np.squeeze(action, axis=0)
        return Act(action=np.asarray(action, dtype=np.float32))

    def reset(self, obs: Obs, instruction: Any, **kwargs) -> dict[str, Any]:
        info = super().reset(obs, instruction, **kwargs)
        self.policy.reset()
        return info


class ManiFlowPolicy(Agent):

    def __init__(
        self,
        default_checkpoint_path: str = "",
        device: str = "cuda:0",
        execution_horizon: int = 1,
        rename_map: dict[str, str] | None = None,
        state_key: str | None = None,
        unnorm_key: str | None = None,
        return_normalized: bool = False,
        apply_action_mode: bool = True,
        use_bfloat16: bool = False,
        include_instruction: bool | None = None,
        num_ddim_steps: int | None = None,
        **kwargs,
    ) -> None:
        super().__init__(default_checkpoint_path=default_checkpoint_path, **kwargs)
        self.device = device
        self.execution_horizon = execution_horizon
        self.rename_map = rename_map or {}
        self.state_key = state_key
        self.unnorm_key = unnorm_key
        self.return_normalized = return_normalized
        self.apply_action_mode = apply_action_mode
        self.use_bfloat16 = use_bfloat16
        self.include_instruction = include_instruction
        self.num_ddim_steps = num_ddim_steps
        self.path = self.checkpoint_path or self.default_checkpoint_path
        if self.checkpoint_step is not None:
            self.path = self.path.format(checkpoint_step=self.checkpoint_step)
        self._cached_actions: deque[np.ndarray] = deque()

    @staticmethod
    def _ensure_hvla_on_path() -> None:
        hvla_root = Path(__file__).resolve().parents[3] / "blocksuite" / "baselines" / "hvla"
        if not hvla_root.exists():
            raise FileNotFoundError(f"Could not locate HVLA source tree at {hvla_root}")
        hvla_root_str = str(hvla_root)
        if hvla_root_str not in sys.path:
            sys.path.insert(0, hvla_root_str)

    @staticmethod
    def _to_chw(array: np.ndarray, *, scale: bool) -> np.ndarray:
        array = np.asarray(array, dtype=np.float32)
        if array.ndim == 3 and array.shape[0] not in (1, 3):
            array = np.moveaxis(array, -1, 0)
        if scale and array.max(initial=0.0) > 1.0:
            array = array / 255.0
        return array

    def initialize(self):
        import torch

        self._ensure_hvla_on_path()
        from hvla.model.framework.base_framework import baseframework

        self.model = baseframework.from_pretrained(self.path).to(self.device).eval()
        if self.use_bfloat16:
            self.model = self.model.to(torch.bfloat16)

        framework_cfg = getattr(self.model.config, "framework", None)
        self.framework_name = getattr(framework_cfg, "name", self.model.__class__.__name__)
        shape_meta = framework_cfg.get("shape_meta", {}) if framework_cfg is not None else {}
        obs_meta = shape_meta.get("obs", {})
        self.state_key = self.state_key or ("agent_pos" if "agent_pos" in obs_meta else None)
        datasets_cfg = getattr(self.model.config, "datasets", None)
        vla_data = getattr(datasets_cfg, "vla_data", None) if datasets_cfg is not None else None
        self.action_mode = vla_data.get("action_mode", "abs") if vla_data is not None else "abs"
        self.language_conditioned = bool(
            framework_cfg is not None and framework_cfg.get("language_conditioned", False)
        )

    def _get_state(self, obs: Obs) -> np.ndarray | None:
        state = obs.state
        if state is None and self.state_key is not None:
            state = obs.info.get(self.state_key)
        if state is None:
            state = obs.info.get("state", obs.info.get("joints"))
        if state is None:
            return None
        return np.asarray(state, dtype=np.float32)

    def _build_obs_dict(self, obs: Obs) -> dict[str, np.ndarray | list[str]]:
        obs_dict: dict[str, np.ndarray | list[str]] = {}
        for source_key, value in obs.cameras.items():
            key = self.rename_map.get(source_key, source_key)
            array = np.array(value, copy=True)
            scale = key.endswith("_rgb")
            if array.ndim == 3:
                array = self._to_chw(array, scale=scale)
            else:
                array = np.asarray(array, dtype=np.float32)
            obs_dict[key] = array[None, None, ...]

        state = self._get_state(obs)
        if state is not None and self.state_key is not None:
            obs_dict[self.state_key] = state[None, None, ...]

        if self.include_instruction is True or (self.include_instruction is None and self.language_conditioned):
            obs_dict["task_name"] = [self.instruction]
        return obs_dict

    @staticmethod
    def _extract_actions(result: Any) -> np.ndarray:
        if isinstance(result, dict):
            result = result.get("normalized_actions", result.get("action", result))
        if hasattr(result, "detach"):
            result = result.detach().cpu().float().numpy()
        actions = np.asarray(result, dtype=np.float32)
        if actions.ndim == 3:
            return actions[0]
        if actions.ndim == 2:
            return actions
        if actions.ndim == 1:
            return actions[None, :]
        raise ValueError(f"Unsupported action output shape {actions.shape}")

    def _denormalize_actions(self, normalized_actions: np.ndarray, obs: Obs) -> np.ndarray:
        if self.return_normalized:
            return normalized_actions

        actions = self.model.unnormalize_actions(
            normalized_actions.copy(),
            self.model.get_action_stats(unnorm_key=self.unnorm_key),
        ).astype(np.float32)
        if not self.apply_action_mode:
            return actions

        state = self._get_state(obs)
        if state is None or state.shape[-1] != actions.shape[-1]:
            return actions
        if self.action_mode == "rel":
            return actions + state[None, :]
        if self.action_mode == "delta":
            out = np.zeros_like(actions)
            out[0] = actions[0] + state
            for idx in range(1, len(actions)):
                out[idx] = actions[idx] + out[idx - 1]
            return out
        return actions

    def _predict_chunk(self, obs: Obs) -> tuple[np.ndarray, np.ndarray]:
        import torch

        obs_dict = self._build_obs_dict(obs)
        with torch.inference_mode():
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.use_bfloat16 and "cuda" in self.device):
                if hasattr(self.model, "predict_action"):
                    kwargs = {"examples": {"obs": obs_dict}}
                    if self.num_ddim_steps is not None:
                        kwargs["num_ddim_steps"] = self.num_ddim_steps
                    result = self.model.predict_action(**kwargs)
                elif hasattr(self.model, "policy") and hasattr(self.model.policy, "predict_action"):
                    result = self.model.policy.predict_action(obs_dict)
                else:
                    raise AttributeError(f"{self.framework_name} does not expose a supported predict_action interface")

        normalized = self._extract_actions(result)
        return self._denormalize_actions(normalized, obs), normalized

    def act(self, obs: Obs) -> Act:
        super().act(obs)
        if self._cached_actions:
            return Act(action=self._cached_actions.popleft().astype(np.float32), done=False, info={})

        action_chunk, normalized_chunk = self._predict_chunk(obs)
        horizon = max(1, min(self.execution_horizon, len(action_chunk)))
        for action in action_chunk[1:horizon]:
            self._cached_actions.append(np.asarray(action, dtype=np.float32))
        return Act(
            action=np.asarray(action_chunk[0], dtype=np.float32),
            done=False,
            info={
                "action_chunk": action_chunk,
                "normalized_action_chunk": normalized_chunk,
                "framework_name": self.framework_name,
            },
        )

    def reset(self, obs: Obs, instruction: Any, **kwargs) -> dict[str, Any]:
        info = super().reset(obs, instruction, **kwargs)
        self._cached_actions.clear()
        return info


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
            "./", self.model_name, source="local", pretrained=True  # root of the vjepa source code  # model type
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

    def act(self, obs: Obs) -> Act:
        # torch imports
        import torch
        from torchvision.io import decode_jpeg

        super().act(obs)

        with torch.no_grad():

            # read from camera-stream
            side = obs.cameras["rgb_side"]

            # [3, 720, 1280]  -> [1, 720, 1280, 3] i.e, [T, C, Patches, dim]
            side = torch.permute(side, (1, 2, 0)).unsqueeze(0)

            # [1, 720, 1280, 3] -> [1, 3, 1, 256, 1408] i.e, [B, C, T, Patches, dim]
            input_image_tensor = (self.transform(side)[None, :]).to(
                device=self.device, dtype=torch.float, non_blocking=True
            )
            # Pre-trained VJEPA 2 ENCODER: [1, 3, 1, 256, 1408] -> [1, 256, 1408]
            z_n = self.world_model.encode(input_image_tensor)

            # [1, 7] -> [B, state_dim]
            # TODO: check gripper state convention
            # in DROID: 0: is close to 0.86: is open?
            # In rcs 0: is close and 1: is open
            s_n = (
                torch.tensor((np.concatenate(([obs.info["xyzrpy"], [1 - obs.gripper]]), axis=0)))  # [1-obs.gripper]
                .unsqueeze(0)
                .to(self.device, dtype=torch.float, non_blocking=True)
            )

            # Action conditioned predictor and zero-shot action inference with CEM
            actions = self.world_model.infer_next_action(z_n, s_n, self.goal_rep)  # [rollout_horizon, 7]

            first_action = actions[0].cpu()
            first_action[-1] = 1 - first_action[-1]

        return Act(action=np.array(first_action))

    def reset(self, obs: Obs, instruction: Any, **kwargs) -> dict[str, Any]:
        super().reset(obs, instruction, **kwargs)
        # imports
        import torch

        img = Image.open(self.goal_img)

        # time dim exp
        goal_image = np.expand_dims(np.array(img), axis=0)
        # batch dim exp
        goal_image_tensor = torch.tensor(self.transform(goal_image)[None, :]).to(
            device=self.device, dtype=torch.float, non_blocking=True
        )

        with torch.no_grad():
            self.goal_rep = self.world_model.encode(goal_image_tensor)

        return {}


class OpenPiModel(Agent):

    def __init__(
        self,
        train_config_name: str = "pi0_droid",
        default_checkpoint_path: str = "gs://openpi-assets/checkpoints/pi0_droid",
        execution_horizon=20,
        **kwargs,
    ) -> None:
        super().__init__(default_checkpoint_path=default_checkpoint_path, **kwargs)
        from openpi.training import config

        logging.info(f"checkpoint_path: {self.checkpoint_path}, checkpoint_step: {self.checkpoint_step}")
        self.openpi_path = self.checkpoint_path.format(checkpoint_step=self.checkpoint_step)

        self.cfg = config.get_config(train_config_name)
        self.execution_horizon = execution_horizon

        self.chunk_counter = self.execution_horizon
        self._cached_action_chunk = None

    def initialize(self):
        from openpi.policies import policy_config
        from openpi.shared import download

        checkpoint_dir = download.maybe_download(self.openpi_path)

        # Create a trained policy.
        self.policy = policy_config.create_trained_policy(self.cfg, checkpoint_dir)

    def act(self, obs: Obs) -> Act:
        super().act(obs)
        observation = {f"observation/{k}": np.copy(v).transpose(2, 0, 1) for k, v in obs.cameras.items()}
        observation.update(
            {
                # openpi expects 0 as gripper open and 1 as closed
                "observation/joint_position": obs.state[:-1],
                "observation/gripper_position": 1 - obs.state[-1],
                "prompt": self.instruction,
            }
        )
        action_chunk = self.policy.infer(observation)["actions"]

        # convert gripper action into vlagents format
        action_chunk[:, -1] = 1 - action_chunk[:, -1]
        self._cached_action_chunk = action_chunk

        return Act(action=action_chunk)

    def reset(self, obs: Obs, instruction: Any):
        super().reset(obs, instruction)
        self.chunk_counter = self.execution_horizon
        self._cached_action_chunk = None
        return {}


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
        # Parse payload components
        assert obs.cameras["rgb_side"].shape == (256, 256, 3), "wrong shape, use lanczos"
        image = obs.cameras["rgb_side"]
        unnorm_key = self.unnorm_key

        # Run VLA Inference
        prompt = self.get_openvla_prompt(self.instruction, self.openvla_path)
        inputs = self.processor(prompt, Image.fromarray(image).convert("RGB")).to(self.device, dtype=torch.bfloat16)
        # to use temperature use: do_sample=True, temperature=50.0
        action = self.vla.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)
        # unsqueeze to add horizon dimension
        return Act(action=action[None])


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

        self.horizon = self.horizon
        self.history = deque(maxlen=self.horizon)
        self.num_obs = 0

        logging.info("==========================")
        logging.info(self.model.dataset_statistics.keys())
        self.policy_fn = supply_rng(
            partial(
                self.model.sample_actions,
                unnormalization_statistics=reduce(getitem, self.unnorm_key, self.model.dataset_statistics)["action"],
            ),
        )
        self.task = None

    def act(self, obs: Obs) -> Act:
        # from octo.model.octo_model import _verify_shapes
        import jax
        from octo.utils.gym_wrappers import stack_and_pad

        super().act(obs)
        assert self.task is not None, "forgot reset?"
        # _verify_shapes(obs, <name>, self.model.example_batch["observation"])

        self.num_obs += 1

        # single image
        assert obs.cameras["rgb_side"].shape == (256, 256, 3), "wrong shape, use lanczos"
        obs = {"image_primary": obs.cameras["rgb_side"]}

        self.history.append(obs)
        assert len(self.history) == self.horizon, "forgot reset?"
        full_obs = stack_and_pad(self.history, self.num_obs)

        actions = self.policy_fn(
            jax.tree_map(
                lambda x: x[None],
                full_obs,
            ),
            self.task,
        )
        # remove the batch dimension (batch, horizon, action)
        return Act(action=np.array(actions[0, :, :]))

    def reset(self, obs: Obs, instruction: Any):
        super().reset(obs, instruction)
        assert obs.cameras["rgb_side"].shape == (256, 256, 3), "wrong shape"
        obs = {"image_primary": obs.cameras["rgb_side"]}
        self.task = self.model.create_tasks(texts=[instruction])
        self.num_obs = 1
        self.history.extend([obs] * self.horizon)
        return {}


class OctoActionDistribution(OctoModel):
    """
    this model does not support history window
    dont use self.step and self.episode as this model is not used sequentially
    """

    def __init__(self, **kwargs) -> None:
        assert kwargs["horizon"] == 1, "horizon must be 1 for OctoActionDistribution"
        super().__init__(**kwargs)

    def act(self, obs: Obs) -> Act:
        """
        Args:
            Obs:
                cameras:
                    rgb_side: np.ndarray[tuple[BATCH, H, W, Literal[3]], np.dtype[np.int8]]
                info:
                    num_samples: int
        Return:
            Act:
                action: None
                info:
                    means: np.ndarray[tuple[BATCH, 7], np.dtype[np.float32]]
                    stds: np.ndarray[tuple[BATCH, 7], np.dtype[np.float32]]
        """
        import jax
        import jax.numpy as jnp

        self._from_shared_memory(obs)

        batch_size = obs.cameras["rgb_side"].shape[0]
        assert obs.cameras["rgb_side"].shape == (batch_size, 256, 256, 3), "wrong shape"
        assert self.instruction is not None, "forgot reset?"
        num_samples = obs.info.get("num_samples", 1)

        x = jnp.array(obs.cameras["rgb_side"])  # BATCH, H, W, 3
        # x_expanded = x[:, None, :, :, :]
        x_expanded = jnp.expand_dims(x, 1)
        x_tiled = jnp.tile(x_expanded, (1, num_samples, 1, 1, 1))  # Shape: [BATCH, N, H, W, 3]
        x_duplicated = x_tiled.reshape(-1, x.shape[1], x.shape[2], x.shape[3])  # Shape: [BATCH*N, H, W, 3]
        full_obs = {
            "image_primary": jnp.expand_dims(x_duplicated, 1),
            "timestep_pad_mask": np.ones((batch_size * num_samples, 1)),
        }
        # full_obs = stack_and_pad(x_duplicated, 1)
        tasks = self.model.create_tasks(texts=[self.instruction] * batch_size * num_samples)
        actions = self.policy_fn(
            full_obs,
            tasks,
        )
        # actions: [num_samples x BATCH, 4, 7]
        # remove the horizon dimension and reshape to [BATCH, num_samples, 7]
        actions = actions[:, 0, :].reshape(batch_size, num_samples, 7)
        stds = jnp.std(actions, axis=1)
        means = jnp.mean(actions, axis=1)

        stds = np.asarray(stds)
        means = np.asarray(means)

        return Act(action=None, info={"means": means, "stds": stds, "actions": np.asarray(actions)})

    def reset(self, obs, instruction):
        self.instruction = instruction
        return {}


class OpenVLADistribution(OpenVLAModel):

    def act(self, obs: Obs) -> Act:
        # no batch dimension here
        import torch

        self._from_shared_memory(obs)

        assert self.instruction is not None, "forgot reset?"
        self.step += 1
        batch_size = obs.cameras["rgb_side"].shape[0]

        # Parse payload components
        images = obs.cameras["rgb_side"]
        actions = []
        unnorm_key = self.unnorm_key
        num_samples = obs.info.get("num_samples", 1)

        # time it
        import time

        t1 = time.time()
        # Run VLA Inference
        prompt = self.get_openvla_prompt(self.instruction, self.openvla_path)

        x_expanded = np.expand_dims(images, 1)
        x_tiled = np.tile(x_expanded, (1, num_samples, 1, 1, 1))  # Shape: [BATCH, N, H, W, 3]
        x_duplicated = x_tiled.reshape(
            -1, images.shape[1], images.shape[2], images.shape[3]
        )  # Shape: [BATCH*N, H, W, 3]

        for image in x_duplicated:
            inputs = self.processor(prompt, Image.fromarray(image).convert("RGB")).to(self.device, dtype=torch.bfloat16)
            # to use temperature use: do_sample=True, temperature=50.0
            action = self.vla.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)
            actions.append(action)

        t2 = time.time()
        logging.info(f"needed time for {len(actions)} was {t2-t1}s")

        # unsqueeze to add horizon dimension
        actions = np.stack(actions).astype(np.float32)
        actions = actions.reshape(batch_size, num_samples, 7)
        means = np.mean(actions, axis=1).astype(np.float32)
        stds = np.std(actions, axis=1).astype(np.float32)
        return Act(action=None, info={"means": means, "stds": stds, "actions": actions})


AGENTS = dict(
    test=TestAgent,
    octo=OctoModel,
    lerobot=LeRobotPolicy,
    maniflow=ManiFlowPolicy,
    openvla=OpenVLAModel,
    octodist=OctoActionDistribution,
    openvladist=OpenVLADistribution,
    openpi=OpenPiModel,
    vjepa=VjepaAC,
)
