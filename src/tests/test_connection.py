import subprocess
from time import sleep

import numpy as np

from vlagents.client import RemoteAgent
from vlagents.eval import start_server
from vlagents.policies.interface import Obs, SingleObs


def _make_obs(data: np.ndarray, instruction: str = "do something") -> Obs:
    return Obs(
        obs={"right": SingleObs(cameras={"rgb_side": data})},
        language_instruction=instruction,
    )


def _single_robot_action_info(act):
    step = act.acts[0]
    robot_action = step["right"]
    return robot_action.action, robot_action.gripper, robot_action.done, robot_action.info


def _test_connection(agent: RemoteAgent):
    data = np.zeros((256, 256, 3), dtype=np.uint8)
    data[2, 0, 0] = 16

    first = agent.act(_make_obs(data))
    first_action, first_gripper, first_done, first_info = _single_robot_action_info(first)
    assert first_info["shapes"] == {"rgb_side": [224, 224, 3]}
    assert first_info["dtype"] == {"rgb_side": "uint8"}
    assert first_info["data"]["rgb_side"].shape == (224, 224, 3)
    assert np.all(first_action == np.array([0, 0, 0, 0, 0, 0], dtype=np.float32))
    assert first_gripper == 0.0
    assert not first_done

    data[0, 0, 2] = 1
    second = agent.act(_make_obs(data))
    second_action, second_gripper, second_done, second_info = _single_robot_action_info(second)
    assert second_info["shapes"] == {"rgb_side": [224, 224, 3]}
    assert second_info["dtype"] == {"rgb_side": "uint8"}
    assert second_info["data"]["rgb_side"].shape == (224, 224, 3)
    assert np.all(second_action == np.array([0, 0, 0, 0, 0, 0], dtype=np.float32))
    assert second_gripper == 1.0
    assert not second_done


def _test_connection_jpeg(agent: RemoteAgent):
    data = np.zeros((256, 256, 3), dtype=np.uint8)
    act = agent.act(_make_obs(data))
    action, gripper, done, info = _single_robot_action_info(act)
    assert info["shapes"] == {"rgb_side": [224, 224, 3]}
    assert info["dtype"] == {"rgb_side": "uint8"}
    assert info["data"]["rgb_side"].shape == (224, 224, 3)
    assert np.all(action == np.array([0, 0, 0, 0, 0, 0], dtype=np.float32))
    assert gripper == 0.0
    assert not done


def _test_connection_without_resize(agent: RemoteAgent):
    data = np.zeros((256, 256, 3), dtype=np.uint8)
    data[2, 0, 0] = 16
    act = agent.act(_make_obs(data))
    _, _, _, info = _single_robot_action_info(act)
    assert info["shapes"] == {"rgb_side": [256, 256, 3]}
    assert info["dtype"] == {"rgb_side": "uint8"}
    np.testing.assert_array_equal(info["data"]["rgb_side"], data)


def test_connection_numpy_serialization():
    with start_server("test", {}, 8080, "localhost") as p:
        sleep(2)
        agent = RemoteAgent("localhost", 8080, "test")
        with agent:
            while not agent.is_initialized():
                sleep(0.1)
            _test_connection(agent)
        p.send_signal(subprocess.signal.SIGINT)


def test_connection_numpy_shm():
    with start_server("test", {}, 8080, "localhost") as p:
        sleep(2)
        agent = RemoteAgent("localhost", 8080, "test", on_same_machine=True)
        with agent:
            while not agent.is_initialized():
                sleep(0.1)
            _test_connection(agent)
        p.send_signal(subprocess.signal.SIGINT)


def test_connection_numpy_jpeg():
    with start_server("test", {}, 8080, "localhost") as p:
        sleep(2)
        agent = RemoteAgent("localhost", 8080, "test", jpeg_encoding=True)
        with agent:
            while not agent.is_initialized():
                sleep(0.1)
            _test_connection_jpeg(agent)
        p.send_signal(subprocess.signal.SIGINT)


def test_connection_preserves_native_resolution():
    with start_server("test", {}, 8080, "localhost") as p:
        sleep(2)
        agent = RemoteAgent("localhost", 8080, "test", image_size=None)
        with agent:
            while not agent.is_initialized():
                sleep(0.1)
            _test_connection_without_resize(agent)
        p.send_signal(subprocess.signal.SIGINT)
