
import numpy as np
from PIL import Image
from vlagents import register_agent
from vlagents.policies.interface import Act, Agent, Obs

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

register_agent("vjepa_ac", VjepaAC)