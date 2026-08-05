import base64
from dataclasses import dataclass, field
from multiprocessing import resource_tracker, shared_memory
from typing import Any

import numpy as np
import simplejpeg

from vlagents import register_agent


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
    # translation in m and rotation around x, y, z (roll, pitch, yaw) axes in radians
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


register_agent("test", TestAgent)
