import logging
import os
import time
import typing
from dataclasses import asdict
from tempfile import TemporaryDirectory
from threading import Thread

import json_numpy
import rpyc

from vlagents.client import dataclass_from_dict
from vlagents.policies import Agent, CameraDataType, Obs, SharedMemoryPayload

logging.basicConfig(
    format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)


@rpyc.service
class AgentService(rpyc.Service):
    GIT_ID = "git_id_remote.txt"
    GIT_ID_SUBMODULES = "git_id_submodules_remote.txt"
    GIT_DIFF = "git_diff_remote.txt"

    def __init__(self, agent: Agent, name: str) -> None:
        super().__init__()
        logging.info("start server")
        self.agent = agent
        self._name = name
        self._is_initialized = False
        # start initialize in thread
        self._init_thread = Thread(target=self._initialize)
        self._init_thread.start()

    def _initialize(self):
        # record time
        logging.info("start heavy init steps")
        t1 = time.time()
        self.agent.initialize()
        t2 = time.time()
        self._is_initialized = True
        print(f"AgentService initialized with {self._name} after {round(t2 - t1)} seconds")
        logging.info(f"AgentService initialized with {self._name} after {round(t2 - t1)} seconds")

    @rpyc.exposed
    def act(self, obs_bytes: bytes) -> str:
        assert self._is_initialized, "AgentService not initialized, wait until is_initialized is True"
        # action, done, info
        obs = typing.cast(Obs, dataclass_from_dict(Obs, json_numpy.loads(obs_bytes)))
        for single_obs in obs.obs.values():
            if single_obs.camera_data_type == CameraDataType.SHARED_MEMORY:
                single_obs.cameras = {
                    camera_name: dataclass_from_dict(SharedMemoryPayload, camera_data)
                    for camera_name, camera_data in single_obs.cameras.items()
                }
        if obs.goal_image_data_type == CameraDataType.SHARED_MEMORY and obs.goal_image is not None:
            obs.goal_image = dataclass_from_dict(SharedMemoryPayload, obs.goal_image)
        return json_numpy.dumps(asdict(self.agent.act(obs)))

    @rpyc.exposed
    def name(self) -> str:
        return self._name

    @rpyc.exposed
    def is_initialized(self) -> bool:
        return self._is_initialized

    @rpyc.exposed
    def git_status(self) -> str:
        with TemporaryDirectory() as tmp_dir:
            # git commit has id
            os.system(f'git log --format="%H" -n 1 > {os.path.join(tmp_dir, self.GIT_ID)}')
            # submodule git ids
            os.system(f"git submodule status > {os.path.join(tmp_dir, self.GIT_ID_SUBMODULES)}")
            # get git diff
            os.system(f"git diff --submodule=diff > {os.path.join(tmp_dir, self.GIT_DIFF)}")

            def read_file(file_path: str) -> str:
                with open(file_path) as f:
                    return f.read()

            return json_numpy.dumps(
                {
                    fn: read_file(os.path.join(tmp_dir, fn))
                    for fn in [self.GIT_ID, self.GIT_ID_SUBMODULES, self.GIT_DIFF]
                }
            )

    def __enter__(self):
        pass

    def __exit__(self, *args, **kwargs):
        self.close()

    def close(self):
        self.agent.close()

    def on_disconnect(self, conn):
        self.close()
