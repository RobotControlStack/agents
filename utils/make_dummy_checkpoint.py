"""Generate a randomly-initialized ("dummy") ManiFlow checkpoint.

This exists purely to smoke-test the serve + sim inference plumbing end to end
(vlagents ``ManiFlowPolicy`` -> ``load_safetensors_policy`` -> ``ManiFlowInferAdapter``
-> rcs sim env). The weights are (mostly) random, so rollouts will be meaningless --
the point is that the checkpoint *loads* and produces correctly-shaped actions.

The architecture is taken verbatim from the real training config
``examples/Franka/train_maniflow_multimodal.yaml`` so the dummy matches production
(DiTX sizes, injection modes, shape_meta, ``action_mode``). The **pointmap modality
is stripped** (RGB-only): the rcs sim / franka.py inference pipeline provides colour
images but no dense XYZ pointmaps, so the dummy is built without the ptmap encoder --
exactly what ``load_safetensors_policy`` does when ``include_ptmap=false``.

It writes a run directory laid out how ``load_safetensors_policy`` discovers things::

    <out>/
      config.yaml                 # the (ptmap-stripped) framework + datasets.vla_data
      stats.json                  # q01/q99 norm stats read by ManiFlowInferAdapter
      dataset_statistics.json     # same payload; written for parity with real runs
      checkpoints/
        dummy_model.safetensors   # framework state_dict (policy.* + ema.*)

The policy is built with hvla's own factory (``ManiFlowMultiModalFramework``) using
the same config the loader re-reads, so the rebuilt architecture matches and
``load_state_dict(strict=True)`` succeeds. The RGB backbone (timm CLIP ViT-B) is
instantiated -- and downloaded on first run, then cached -- because the loader does
the same.

Usage::

    python -m blocksuite.serving.make_dummy_checkpoint --out /tmp/dummy_maniflow

Then serve it through the vlagents server that franka.py talks to::

    python -m vlagents start-server maniflow --port 20000 --kwargs \
        '{"checkpoint_path": "/tmp/dummy_maniflow/checkpoints/dummy_model.safetensors", \
          "stats_path": "/tmp/dummy_maniflow/stats.json"}'
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np

_HVLA_ROOT = Path(__file__).resolve().parents[2] / "baselines" / "hvla"
DEFAULT_CONFIG = _HVLA_ROOT / "examples" / "Franka" / "train_maniflow_multimodal.yaml"

# Nominal FR3 joint limits, used as plausible q01/q99 for the state stats.
FR3_Q_MIN = [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]
FR3_Q_MAX = [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]
REL_DELTA = 0.1


def load_config(config_path: Path):
    """Load the training config and strip the pointmap (RGB-only) modality.

    Mirrors ``load_safetensors_policy``'s ``include_ptmap=false`` handling so the
    saved config re-builds an identical RGB-only policy at serve time.
    """
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(str(config_path))
    OmegaConf.set_struct(cfg, False)  # allow adding use_ptmap if absent

    OmegaConf.update(cfg, "framework.use_ptmap", False)
    OmegaConf.update(cfg, "datasets.vla_data.include_ptmap", False)

    obs = cfg.framework.shape_meta.obs
    for key in [k for k in obs if str(k).endswith("_ptmap")]:
        del obs[key]

    return cfg


def config_dims(cfg) -> dict:
    """Pull the shapes/mode the stats + verify obs need out of the config."""
    from omegaconf import OmegaConf

    sm = OmegaConf.to_container(cfg.framework.shape_meta, resolve=True)
    rgb_keys = sorted(k for k in sm["obs"] if k.endswith("_rgb"))
    action_mode_raw = str(OmegaConf.select(cfg, "datasets.vla_data.action_mode", default="abs"))
    return {
        "rgb_keys": rgb_keys,
        "img_size": sm["obs"][rgb_keys[0]]["shape"][-1],
        "state_dim": sm["obs"]["agent_pos"]["shape"][0],
        "action_dim": sm["action"]["shape"][0],
        "is_relative": action_mode_raw.startswith("rel"),
    }


def build_stats(is_relative: bool) -> dict:
    """Norm stats in the flat schema ManiFlowInferAdapter._init_stats reads."""
    action_joint_key = (
        "action.relative_joint_target" if is_relative else "action.joint_target"
    )
    action_joint_range = (
        {"q01": [-REL_DELTA] * 7, "q99": [REL_DELTA] * 7}
        if is_relative
        else {"q01": FR3_Q_MIN, "q99": FR3_Q_MAX}
    )
    return {
        "observation.state.joint_position": {"q01": FR3_Q_MIN, "q99": FR3_Q_MAX},
        "observation.state.gripper_position": {"q01": 0.0, "q99": 1.0},
        action_joint_key: action_joint_range,
        "action.gripper_position": {"q01": 0.0, "q99": 1.0},
        "action.pd_mode": {"q01": 0.0, "q99": 1.0},
    }


def save_state_dict(framework, path: Path) -> None:
    from safetensors.torch import save_file

    # clone + contiguous so safetensors never sees shared/viewed storage.
    tensors = {k: v.detach().cpu().contiguous().clone() for k, v in framework.state_dict().items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(path))
    n_policy = sum(1 for k in tensors if k.startswith("policy."))
    logging.info("Saved %d tensors (%d under policy.*) to %s", len(tensors), n_policy, path)


def verify(checkpoint: Path, stats_path: Path, dims: dict) -> None:
    """Reload through the real serve path and run one inference."""
    from blocksuite.serving.serve_maniflow_policy import Args, load_safetensors_policy

    # action_mode=None -> auto-detected from the saved config.yaml.
    adapter = load_safetensors_policy(Args(checkpoint=str(checkpoint), stats_path=str(stats_path)))
    img = dims["img_size"]
    obs: dict = {k: np.zeros((img, img, 3), dtype=np.uint8) for k in dims["rgb_keys"]}
    obs["agent_pos"] = np.zeros(dims["state_dim"], dtype=np.float32)
    out = adapter.infer(obs)
    actions = np.asarray(out["actions"])
    logging.info("Verify OK: infer() returned actions with shape %s", actions.shape)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True, type=Path, help="Output run directory.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Training config to take the architecture from.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-verify", action="store_true", help="Skip the reload + infer check.")
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU-only (hides the GPU). Use when the installed torch build has "
        "no kernels for this GPU (CUDA error: no kernel image is available).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # Must happen before torch is imported so CUDA is never initialized; this makes
    # torch.cuda.is_available() return False, so the verify adapter runs on CPU.
    if args.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    import torch
    from omegaconf import OmegaConf

    from hvla.model.framework.ManiFlowMultiModal import ManiFlowMultiModalFramework

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    logging.info("Loading architecture from %s (stripping pointmap modality; RGB-only)", args.config)
    cfg = load_config(args.config)
    dims = config_dims(cfg)
    logging.info(
        "RGB cams=%s img=%d state_dim=%d action_dim=%d action_mode=%s",
        dims["rgb_keys"], dims["img_size"], dims["state_dim"], dims["action_dim"],
        "relative" if dims["is_relative"] else "absolute",
    )

    out: Path = args.out
    ckpt_path = out / "checkpoints" / "dummy_model.safetensors"
    config_path = out / "config.yaml"
    stats_path = out / "stats.json"
    dataset_stats_path = out / "dataset_statistics.json"

    logging.info("Building ManiFlow framework (random flow head); this instantiates/downloads the RGB backbone...")
    framework = ManiFlowMultiModalFramework(config=cfg).eval()

    out.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, config_path)
    logging.info("Wrote config to %s", config_path)

    stats = build_stats(dims["is_relative"])
    stats_path.write_text(json.dumps(stats, indent=2))
    dataset_stats_path.write_text(json.dumps(stats, indent=2))
    logging.info("Wrote stats to %s (and %s)", stats_path, dataset_stats_path)

    save_state_dict(framework, ckpt_path)

    if not args.no_verify:
        # Free the build before rebuilding through the serve path.
        del framework
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        verify(ckpt_path, stats_path, dims)

    logging.info(
        "Done. Serve with:\n"
        "  python -m vlagents start-server maniflow --port 20000 --kwargs '%s'",
        json.dumps({"checkpoint_path": str(ckpt_path), "stats_path": str(stats_path)}),
    )


if __name__ == "__main__":
    main()
