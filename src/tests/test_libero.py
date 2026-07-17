if __name__ == "__main__":
    import datetime
    import os

    import numpy as np
    from PIL import Image

    from lerobot.envs.libero import LiberoEnv
    from vlagents.__main__ import _run_eval
    from vlagents.evaluator_envs import AgentConfig, EvalConfig

    # main_app()
    # test
    os.environ["RUN_PATH"] = "test_output"
    _run_eval(
        output_path="test_output",
        eval_cfgs=[
            EvalConfig(
                env_id="libero_10",
                env_kwargs={"controller": "OSC_POSE", "camera_heights": 256, "camera_widths": 256},
                max_steps_per_episode=100,
            )
        ],
        agent_cfg=AgentConfig(host="localhost", port=8080, agent_name="test", agent_kwargs={}),
        n_processes=1,
        n_gpus=1,
        episodes=1,
    )

    # from libero.libero import benchmark, get_libero_path
    # from libero.libero.envs import OffScreenRenderEnv
    # benchmark_dict = benchmark.get_benchmark_dict()

    # task_suite = benchmark_dict["libero_10"]()
    # env = LiberoEnv(task_suite, task_id=0, task_suite_name="libero_10")
    # env.reset()
    # im = []
    # im2 = []
    # for i in range(100):
    #     obs, reward, done, truncated, info = env.step(np.zeros(7))
    #     im2.append(Image.fromarray(obs["pixels"]["image"]))
    #     im.append(Image.fromarray(obs["pixels"]["image2"]))
    # env.close()

    # im[0].save(
    #     f"{str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))}.gif",
    #     save_all=True,
    #     append_images=im[1:],
    #     duration=0.2 * 1000,
    #     loop=0,
    # )
