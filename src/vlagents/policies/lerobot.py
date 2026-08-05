
import numpy as np
from vlagents import register_agent
from vlagents.policies.interface import Act, Agent, Obs

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

register_agent("lerobot", LeRobotPolicy)