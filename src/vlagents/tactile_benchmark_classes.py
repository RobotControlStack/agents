from vlagents.policies import Agent, AGENTS
from vlagents.evaluator_envs import Obs, Act
from typing import Any
import numpy as np
import logging
from copy import deepcopy

class TactileBenchmarkAgent(Agent):

    def __init__(
        self,
        policy_name: str = "act-tact-bench",
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

        self.rename_map = {
            "side": "side",
            "wrist": "wrist",
            "digit_right_right": "digit_right_right",
            "digit_right_left": "digit_right_left",
        }

    def initialize(self):
        from collections import deque
        import importlib

        import torch
        from lerobot.configs import PreTrainedConfig
        from lerobot.policies.factory import get_policy_class, make_pre_post_processors
        from torchvision.transforms import v2
        import tactile_pipeline.policies.act.configuration_act
        import tactile_pipeline.policies.act.modeling_act

        # from vlagents import train_xvla

        policy_config = PreTrainedConfig.from_pretrained(self.path)
        policy_class = get_policy_class(policy_config.type)
        self.policy = policy_class.from_pretrained(self.path, config=policy_config, strict=True)
        self.policy.config.n_action_steps = self.n_action_steps
        logging.info(
            "Loaded policy: type=%s variant=%s class=%s checkpoint=%s device=%s chunk_size=%s n_action_steps=%s "
            "temporal_ensemble_coeff=%s freeze_variant_backbone=%s image_keys=%s",
            policy_config.type,
            getattr(self.policy.config, "act_variant", "n/a"),
            self.policy.__class__.__name__,
            self.path,
            self.device,
            getattr(self.policy.config, "chunk_size", "n/a"),
            self.policy.config.n_action_steps,
            getattr(self.policy.config, "temporal_ensemble_coeff", "n/a"),
            getattr(self.policy.config, "freeze_variant_backbone", "n/a"),
            [
                key.removeprefix("observation.images.")
                for key in self.policy.config.input_features
                if key.startswith("observation.images.")
            ],
        )
        self.policy_input_image_keys = [key.removeprefix("observation.images.") for key in self.policy.config.input_features if key.startswith("observation.images")]
        if getattr(self.policy, "name", None) == "act":
            policy_module = importlib.import_module(self.policy.__class__.__module__)
            ACTTemporalEnsembler = getattr(policy_module, "ACTTemporalEnsembler")

            if self.temporal_ensemble_coeff is not None:
                print("Temporal ensembling will be used")
                self.policy.config.temporal_ensemble_coeff = self.temporal_ensemble_coeff
                self.policy.temporal_ensembler = ACTTemporalEnsembler(
                    self.temporal_ensemble_coeff,
                    self.policy.config.chunk_size,
                )
            elif self.policy.config.temporal_ensemble_coeff is None:
                print("No temporal ensembling will be used")
                if hasattr(self.policy, "temporal_ensembler"):
                    delattr(self.policy, "temporal_ensembler")
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
            if key not in self.policy_input_image_keys:
                continue
            expected_shape = self._expected_image_shapes.get(self.rename_map.get(key, key))
            assert expected_shape is not None, f"Unexpected camera key: {key}. Expected keys: {list(self._expected_image_shapes.keys())}"
            observation[f"observation.images.{self.rename_map.get(key, key)}"] = self._camera_transforms[
                self.rename_map.get(key, key)
            ](np.array(img_data, copy=True))
        observation = self.preprocessor(observation)
        with torch.inference_mode():
            action = self.policy.select_action(observation)
            # action = self.policy.predict_action_chunk(observation)
            action_raw = deepcopy(action)
        action = self.postprocessor(action)

        if isinstance(action, torch.Tensor):
            action = action.detach().float().cpu().numpy()

        action = np.squeeze(action, axis=0)
        # Home pose
        return Act(action=np.asarray(action, dtype=np.float32))

    def reset(self, obs: Obs, instruction: Any, **kwargs) -> dict[str, Any]:
        info = super().reset(obs, instruction, **kwargs)
        self.policy.reset()
        return info

AGENTS["tb_agent"] = TactileBenchmarkAgent
