import torch
import torch.nn as nn
from lerobot.policies.xvla.action_hub import BaseActionSpace, register_action

XVLA_DOMAIN_ID = 20


@register_action("frankaduo")
class FrankaDuoActionSpace(BaseActionSpace):
    """Custom action space for dual Franka setup."""

    dim_action = 20

    # Use lists for safe PyTorch advanced indexing
    gripper_idx = (7, 15)
    # All indices EXCEPT 7 and 15
    joint_idx = [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14]

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()

    def compute_loss(self, pred, target):
        """Define your loss computation."""
        # Corrected: Now computes MSE for BOTH Robot 1 and Robot 2 joints
        joints_loss = self.mse(pred[..., self.joint_idx], target[..., self.joint_idx])

        # Computes BCE for both grippers
        gripper_loss = self.bce(pred[..., self.gripper_idx], target[..., self.gripper_idx])

        return {
            "joints_loss": joints_loss,
            "gripper_loss": gripper_loss,
        }

    def preprocess(self, proprio, action, mode="train"):
        """Preprocess actions before training."""
        proprio_m = proprio.clone()
        action_m = action.clone() if action is not None else None

        # Zero out both grippers
        proprio_m[..., self.gripper_idx] = 0.0
        if action_m is not None:
            action_m[..., self.gripper_idx] = 0.0

        return proprio_m, action_m

    def postprocess(self, action):
        """Post-process predictions for deployment."""
        # Apply sigmoid to both gripper logits
        action[..., self.gripper_idx] = torch.sigmoid(action[..., self.gripper_idx])
        return action[..., :16]
