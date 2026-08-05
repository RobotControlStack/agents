import base64
import dataclasses
from dataclasses import asdict
from multiprocessing import shared_memory
from typing import get_args, get_origin

import json_numpy
import numpy as np
import rpyc
import simplejpeg
from PIL import Image

from vlagents.policies.interface import Act, Agent, CameraDataType, Obs, SharedMemoryPayload


def dataclass_from_dict(klass, value):
    origin = get_origin(klass)
    if origin is dict:
        key_type, value_type = get_args(klass)
        return {
            dataclass_from_dict(key_type, key): dataclass_from_dict(value_type, item) for key, item in value.items()
        }
    if origin is list:
        (item_type,) = get_args(klass)
        return [dataclass_from_dict(item_type, item) for item in value]

    if dataclasses.is_dataclass(klass):
        fieldtypes = {f.name: f.type for f in dataclasses.fields(klass)}
        return klass(**{field: dataclass_from_dict(fieldtypes[field], value[field]) for field in value})

    return value


class RemoteAgent(Agent):
    def __init__(
        self,
        host: str,
        port: int,
        model: str,
        on_same_machine: bool = False,
        jpeg_encoding: bool = False,
        image_size: tuple[int, int] | None = (224, 224),
    ):
        """Connect to a remote agent service.

        Args:
            host (str): Hostname or IP address of the remote agent service.
            port (int): Port number of the remote agent service.
            model (str): Name of the model to connect to.
            on_same_machine (bool, optional): If True, assumes the agent is running on the same machine and uses
                shared memory for more efficient communication. Defaults to False.
            jpeg_encoding (bool, optional): If True the image data is jpeg encoded for smaller transfer size.
                Defaults to False.
            image_size (tuple[int, int] | None, optional): Image size as (width, height) applied before
                serialization. Set to None to retain native resolution. Defaults to (224, 224).
        """
        self.host = host
        self.port = port
        self.model = model
        self.on_same_machine = on_same_machine
        self.jpeg_encoding = jpeg_encoding
        self.image_size = self._validate_image_size(image_size)
        self._shm: dict[str, shared_memory.SharedMemory] = {}
        self.c = None
        self._connect()

    def _connect(self):
        self.c = rpyc.connect(
            self.host,
            self.port,
            config={"allow_pickle": True, "allow_public_attrs": True, "sync_request_timeout": 300},
        )
        assert self.model == self.c.root.name()

    def reconnect(
        self,
        host: str | None = None,
        port: int | None = None,
        model: str | None = None,
        on_same_machine: bool | None = None,
        jpeg_encoding: bool | None = None,
        image_size: tuple[int, int] | None = None,
    ):
        if self.c is not None:
            try:
                self.c.close()
            except Exception:
                pass
        if host is not None:
            self.host = host
        if port is not None:
            self.port = port
        if model is not None:
            self.model = model
        if on_same_machine is not None:
            self.on_same_machine = on_same_machine
        if image_size is not None:
            self.image_size = self._validate_image_size(image_size)
        if jpeg_encoding is not None:
            self.jpeg_encoding = jpeg_encoding
        self._connect()

    def ensure_connected(self):
        try:
            assert self.c is not None
            self.c.ping()
        except Exception:
            self.reconnect()

    def _to_shared_memory_payload(self, shm_key: str, image: np.ndarray) -> SharedMemoryPayload:
        if shm_key not in self._shm or self._shm[shm_key].size < image.nbytes:
            if shm_key in self._shm:
                self._shm[shm_key].close()
                self._shm[shm_key].unlink()
            self._shm[shm_key] = shared_memory.SharedMemory(create=True, size=image.nbytes)
        image_shared = np.ndarray(image.shape, buffer=self._shm[shm_key].buf, dtype=image.dtype)
        image_shared[:] = image[:]
        return SharedMemoryPayload(
            shm_name=self._shm[shm_key].name,
            shape=image.shape,
            dtype=image.dtype.name,
        )

    @staticmethod
    def _to_jpeg_payload(image: np.ndarray) -> str:
        return base64.urlsafe_b64encode(simplejpeg.encode_jpeg(np.ascontiguousarray(image))).decode("utf-8")

    @staticmethod
    def _validate_image_size(image_size: tuple[int, int] | None) -> tuple[int, int] | None:
        if image_size is None:
            return None
        if len(image_size) != 2 or any(not isinstance(size, (int, np.integer)) or size <= 0 for size in image_size):
            message = "image_size must be a (width, height) pair of positive integers or None"
            raise ValueError(message)
        return tuple(int(size) for size in image_size)

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        if self.image_size is None or image.shape[:2] == self.image_size[::-1]:
            return image
        if image.ndim == 3:
            return np.asarray(Image.fromarray(image).resize(self.image_size, Image.Resampling.BILINEAR))
        if image.ndim == 4:
            return np.stack([self._resize_image(frame) for frame in image])
        message = f"Expected an HWC image or NHWC batch, got shape {image.shape}"
        raise ValueError(message)

    def _process(self, obs: Obs) -> Obs:
        for robot_name, single_obs in obs.obs.items():
            single_obs.cameras = {
                camera_name: self._resize_image(camera_data) for camera_name, camera_data in single_obs.cameras.items()
            }
            if self.on_same_machine:
                camera_dict = {}
                for camera_name, camera_data in single_obs.cameras.items():
                    assert isinstance(camera_data, np.ndarray)
                    camera_dict[camera_name] = self._to_shared_memory_payload(
                        f"{robot_name}:{camera_name}", camera_data
                    )
                single_obs.cameras = camera_dict
                single_obs.camera_data_type = CameraDataType.SHARED_MEMORY
            elif self.jpeg_encoding:
                camera_dict = {}
                for camera_name, camera_data in single_obs.cameras.items():
                    assert isinstance(camera_data, np.ndarray)
                    camera_dict[camera_name] = self._to_jpeg_payload(camera_data)
                single_obs.cameras = camera_dict
                single_obs.camera_data_type = CameraDataType.JPEG_ENCODED

        if obs.goal_image is not None:
            assert isinstance(obs.goal_image, np.ndarray)
            obs.goal_image = self._resize_image(obs.goal_image)
            if self.on_same_machine:
                obs.goal_image = self._to_shared_memory_payload("goal_image", obs.goal_image)
                obs.goal_image_data_type = CameraDataType.SHARED_MEMORY
            elif self.jpeg_encoding:
                obs.goal_image = self._to_jpeg_payload(obs.goal_image)
                obs.goal_image_data_type = CameraDataType.JPEG_ENCODED

        return obs

    def act(self, obs: Obs) -> Act:
        obs = self._process(obs)
        obs = json_numpy.dumps(asdict(obs))
        # action, done, info
        try:
            assert self.c is not None
            return dataclass_from_dict(Act, json_numpy.loads(self.c.root.act(obs)))
        except Exception:
            self.reconnect()
            assert self.c is not None
            return dataclass_from_dict(Act, json_numpy.loads(self.c.root.act(obs)))

    def git_status(self) -> str:
        assert self.c is not None
        return json_numpy.loads(self.c.root.git_status())

    def is_initialized(self) -> bool:
        assert self.c is not None
        return self.c.root.is_initialized()

    def close(self):
        for shm in self._shm.values():
            shm.close()
            shm.unlink()
        self._shm = {}
        if self.c is not None:
            self.c.close()


if __name__ == "__main__":
    # to test the connection
    from vlagents.policies.interface import SingleObs

    agent = RemoteAgent("localhost", 8080, "test")
    obs = Obs(
        obs={"right": SingleObs(cameras={"rgb_side": np.zeros((256, 256, 3), dtype=np.uint8)})},
        language_instruction="do something",
    )
    print(agent.act(obs))
    print(agent.act(obs))
