import base64
import dataclasses
from dataclasses import asdict
from multiprocessing import shared_memory
from typing import Any, get_args, get_origin

import json_numpy
import numpy as np
import rpyc
import simplejpeg

from vlagents.policies import Act, Agent, CameraDataType, Obs, SharedMemoryPayload


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
    def __init__(self, host: str, port: int, model: str, on_same_machine: bool = False, jpeg_encoding: bool = False):
        """Connect to a remote agent service.

        Args:
            host (str): Hostname or IP address of the remote agent service.
            port (int): Port number of the remote agent service.
            model (str): Name of the model to connect to.
            on_same_machine (bool, optional): If True, assumes the agent is running on the same machine and uses
                shared memory for more efficient communication. Defaults to False.
            jpeg_encoding (bool, optional): If True the image data is jpeg encoded for smaller transfer size.
                Defaults to False.
        """
        self.host = host
        self.port = port
        self.model = model
        self.on_same_machine = on_same_machine
        self.jpeg_encoding = jpeg_encoding
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
        if jpeg_encoding is not None:
            self.jpeg_encoding = jpeg_encoding
        self._connect()

    def ensure_connected(self):
        try:
            assert self.c is not None
            self.c.ping()
        except Exception:
            self.reconnect()

    def _process(self, obs: Obs) -> Obs:
        for robot_name, single_obs in obs.obs.items():
            if self.on_same_machine:
                camera_dict = {}
                for camera_name, camera_data in single_obs.cameras.items():
                    assert isinstance(camera_data, np.ndarray)
                    shm_key = f"{robot_name}:{camera_name}"
                    if shm_key not in self._shm or self._shm[shm_key].size < camera_data.nbytes:
                        if shm_key in self._shm:
                            self._shm[shm_key].close()
                            self._shm[shm_key].unlink()
                        self._shm[shm_key] = shared_memory.SharedMemory(create=True, size=camera_data.nbytes)
                    camera_shared = np.ndarray(
                        camera_data.shape, buffer=self._shm[shm_key].buf, dtype=camera_data.dtype
                    )
                    camera_shared[:] = camera_data[:]
                    camera_dict[camera_name] = SharedMemoryPayload(
                        shm_name=self._shm[shm_key].name,
                        shape=camera_data.shape,
                        dtype=camera_data.dtype.name,
                    )
                single_obs.cameras = camera_dict
                single_obs.camera_data_type = CameraDataType.SHARED_MEMORY
            elif self.jpeg_encoding:
                camera_dict = {}
                for camera_name, camera_data in single_obs.cameras.items():
                    assert isinstance(camera_data, np.ndarray)
                    camera_dict[camera_name] = base64.urlsafe_b64encode(
                        simplejpeg.encode_jpeg(np.ascontiguousarray(camera_data))
                    ).decode("utf-8")
                single_obs.cameras = camera_dict
                single_obs.camera_data_type = CameraDataType.JPEG_ENCODED
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
    from vlagents.policies import SingleObs

    agent = RemoteAgent("localhost", 8080, "test")
    obs = Obs(
        obs={"right": SingleObs(cameras={"rgb_side": np.zeros((256, 256, 3), dtype=np.uint8)})},
        language_instruction="do something",
    )
    print(agent.act(obs))
    print(agent.act(obs))
