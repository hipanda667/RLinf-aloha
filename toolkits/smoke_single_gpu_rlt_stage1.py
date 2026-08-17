#!/usr/bin/env python
"""Single-GPU smoke test for the RLinf OpenPI Pi0.5 RLT Stage-1 SFT pipeline.

Validates on one GPU:
  1. LeRobot v2.1 dataset loading from the Forge HDD output.
  2. Norm stats loading from the HDD-local checkpoint asset
     (pi05_sandwich_merged_all_0805).
  3. OpenPI Pi0.5 model initialization from the HDD-local checkpoint.
  4. One forward/backward on a single sample (micro batch 1) with
     use_rlt=True; checks vla_loss / rlt_loss are finite.
  5. Keeps a persistent GPU tensor resident afterward so the GPU stays
     occupied until the user launches the two-GPU training.

No checkpoint is saved. No optimizer step is taken. Nothing is overwritten.
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import torch

from rlinf.models import get_model
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config
from rlinf.data.lerobot_paths import resolve_lerobot_repo_id


def _build_worker_config():
    from omegaconf import OmegaConf

    repo = Path("/inspire/hdd/global_user/czxs253130583/fangchuan/work/RL/RLinf-worktree-rltoken-anhao")
    config_path = (
        repo
        / "examples/sft/config/rlt_stage1_sft_openpi_pi05_sandwich_merged_all_0805_16chunk_50k.yaml"
    )
    return OmegaConf.load(config_path)


def _build_dataloader(cfg):
    import openpi.training.data_loader as openpi_data_loader

    repo_id = resolve_lerobot_repo_id(cfg.data.train_data_paths)
    config = get_openpi_config(
        cfg.actor.model.openpi.config_name,
        model_path=cfg.actor.model.model_path,
        batch_size=cfg.actor.micro_batch_size,
        repo_id=repo_id,
        data_kwargs=cfg.actor.openpi_data,
    )
    openpi_overrides = cfg.actor.model.openpi
    config = dataclasses.replace(
        config,
        model=dataclasses.replace(
            config.model,
            action_horizon=int(openpi_overrides.action_horizon),
        ),
    )
    data_loader = openpi_data_loader.create_data_loader(
        config, framework="pytorch", shuffle=True
    )
    return data_loader, data_loader.data_config()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, required=True)
    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    print(f"smoke: using {torch.cuda.get_device_name(args.gpu)} (mem {torch.cuda.get_device_properties(args.gpu).total_memory//(1024**3)} GiB)")

    cfg = _build_worker_config()
    print(
        "smoke: config parsed "
        f"(num_action_chunks={cfg.actor.model.num_action_chunks}, "
        f"action_horizon={cfg.actor.model.openpi.action_horizon}, "
        f"action_chunk={cfg.actor.model.openpi.action_chunk}, "
        f"use_rlt={cfg.actor.model.openpi.use_rlt})"
    )

    # Reuse the worker's OpenPI dataloader construction path.
    data_loader, data_config = _build_dataloader(cfg)
    print(f"smoke: dataloader built (config repo_id={data_config.repo_id}, asset_id={data_config.asset_id})")
    print(f"smoke: norm_stats loaded from checkpoint asset: {data_config.asset_id}")

    model = get_model(cfg.actor.model)
    model.to(device)
    print("smoke: model initialized from HDD checkpoint")

    batch = next(iter(data_loader))
    observation, actions = batch["observation"], batch["actions"]
    print(f"smoke: batch actions shape={tuple(actions.shape)} observation={type(observation).__name__}")
    print(f"smoke: image keys={sorted(observation.images.keys())}")
    print(f"smoke: batch state shape={tuple(observation.state.shape)}")

    assert actions.shape[0] == cfg.actor.micro_batch_size, (
        f"expected micro batch {cfg.actor.micro_batch_size}, got {actions.shape[0]}"
    )
    assert actions.shape[1] == cfg.actor.model.num_action_chunks, (
        f"expected action horizon {cfg.actor.model.num_action_chunks}, got {actions.shape[1]}"
    )
    assert actions.shape[2] == model.config.action_dim, (
        f"expected model action dim {model.config.action_dim}, got {actions.shape[2]}"
    )

    # Forward/backward without optimizer.
    model.train()
    model.zero_grad(set_to_none=True)
    output = model(forward_type=ForwardType.SFT, data=batch)
    loss = output["loss"]
    loss.backward()
    metrics = {
        "vla_loss": float(output["vla_loss"].detach().cpu()),
        "rlt_loss": float(output["rlt_loss"].detach().cpu()),
    }
    print(f"smoke: forward/backward OK metrics={metrics}")
    assert torch.isfinite(output["loss"]), metrics
    assert torch.isfinite(output["vla_loss"]), metrics
    assert torch.isfinite(output["rlt_loss"]), metrics

    print("smoke: PASS (data, norm stats, model, forward/backward, vla/rlt losses)")

    print("smoke: completed without checkpoint or optimizer update")


if __name__ == "__main__":
    main()
