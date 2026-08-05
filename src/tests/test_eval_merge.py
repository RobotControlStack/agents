import numpy as np

from vlagents.__main__ import _merge_env_split_results
from vlagents.envs.interface import EvalConfig


def test_merge_env_split_results_keeps_distinct_seeded_cfgs():
    cfg_a = EvalConfig("duobench/spring_door", {}, max_steps_per_episode=900, seed=0, jpeg_encoding=True)
    cfg_b = EvalConfig("duobench/spring_door", {}, max_steps_per_episode=900, seed=10, jpeg_encoding=True)

    results = [
        (
            np.array([[[1.0, 0.1, 100.0], [0.0, 0.2, 200.0]]]),
            [[[0.1], [0.2]]],
            [0.15],
            40000,
        ),
        (
            np.array([[[0.0, 0.3, 300.0], [1.0, 0.4, 400.0]]]),
            [[[0.3], [0.4]]],
            [0.35],
            40000,
        ),
    ]

    merged_last_reward, merged_rewards, merged_mean_rewards, merged_step = _merge_env_split_results(
        results=results,
        worker_eval_cfgs=[[cfg_a], [cfg_b]],
        eval_cfgs=[cfg_a, cfg_b],
    )

    assert merged_step == 40000
    assert np.array_equal(merged_last_reward[0], results[0][0][0])
    assert np.array_equal(merged_last_reward[1], results[1][0][0])
    assert merged_rewards == [results[0][1][0], results[1][1][0]]
    assert merged_mean_rewards == [0.15, 0.35]
