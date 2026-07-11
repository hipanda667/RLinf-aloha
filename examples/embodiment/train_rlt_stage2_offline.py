# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Offline Stage 2 training entrypoint for RL Token.

This entrypoint:

1. Creates only the Stage 2 actor-critic worker group.
2. Loads an offline RLinf replay-buffer checkpoint through
   OfflineRLTACFSDPPolicy.
3. Runs actor-critic updates without rollout or environment workers.
4. Uses RLinf's existing OfflineRunner for logging, checkpointing,
   and resume support.

Expected worker implementation:

    rlinf/workers/actor/fsdp_offline_rlt_ac_policy_worker.py

Expected configuration:

    examples/embodiment/config/rlt_stage2_offline_ac_mlp.yaml
"""

from __future__ import annotations

import json
import os

import hydra
import torch.multiprocessing as mp
from omegaconf import DictConfig, OmegaConf, open_dict

from rlinf.config import validate_cfg
from rlinf.runners.offline_runner import OfflineRunner
from rlinf.scheduler import Cluster
from rlinf.utils.placement import HybridComponentPlacement
from rlinf.workers.actor.fsdp_offline_rlt_ac_policy_worker import (
    OfflineRLTACFSDPPolicy,
)


mp.set_start_method("spawn", force=True)


def _prepare_offline_cfg(cfg: DictConfig) -> DictConfig:
    """Fill offline-only defaults before RLinf config validation."""

    with open_dict(cfg):
        # RLinf has a dedicated offline task type. Using "embodied" would make
        # validate_cfg expect rollout and environment configurations.
        cfg.runner.task_type = "offline"
        cfg.runner.offline_only = True

        # Required by RLinf's offline configuration validation.
        cfg.runner.local_update_steps = int(
            cfg.runner.get("local_update_steps", 1)
        )
        cfg.runner.log_interval = int(cfg.runner.get("log_interval", 1))

        max_steps = int(cfg.runner.get("max_steps", -1))
        max_epochs = cfg.runner.get("max_epochs", None)

        if max_steps < 0 and max_epochs is None:
            raise ValueError(
                "Offline training requires runner.max_steps >= 0 or "
                "runner.max_epochs to be specified."
            )

        # OfflineRunner accesses max_epochs even when max_steps is used as the
        # effective stopping criterion.
        if max_epochs is None:
            cfg.runner.max_epochs = max_steps
        else:
            cfg.runner.max_epochs = int(max_epochs)

        cfg.runner.only_eval = bool(cfg.runner.get("only_eval", False))
        cfg.runner.val_check_interval = int(
            cfg.runner.get("val_check_interval", -1)
        )
        cfg.runner.save_interval = int(
            cfg.runner.get("save_interval", -1)
        )

    return cfg


def _validate_offline_rlt_cfg(cfg: DictConfig) -> None:
    """Perform RLT-specific checks not covered by generic validation."""

    if cfg.algorithm.loss_type != "rlt_ac":
        raise ValueError(
            "train_rlt_stage2_offline.py requires "
            "algorithm.loss_type='rlt_ac', got "
            f"{cfg.algorithm.loss_type!r}."
        )

    if not bool(cfg.runner.get("offline_only", False)):
        raise ValueError("runner.offline_only must be true.")

    # This entrypoint intentionally creates no environment or rollout workers.
    if cfg.runner.only_eval:
        raise ValueError(
            "Pure offline RLT training does not support runner.only_eval=true."
        )

    if int(cfg.runner.val_check_interval) > 0:
        raise ValueError(
            "Pure offline RLT training has no environment for validation. "
            "Set runner.val_check_interval=-1."
        )

    replay_cfg = cfg.algorithm.get("replay_buffer", None)
    if replay_cfg is None:
        raise ValueError("algorithm.replay_buffer is required.")

    load_path = replay_cfg.get("load_path", None)
    if not load_path:
        raise ValueError(
            "Offline RLT training requires "
            "algorithm.replay_buffer.load_path."
        )

    load_path = os.path.abspath(os.path.expanduser(str(load_path)))
    if not os.path.isdir(load_path):
        raise FileNotFoundError(
            f"Replay-buffer directory does not exist: {load_path}"
        )

    required_buffer_files = (
        "metadata.json",
        "trajectory_index.json",
    )
    missing_files = [
        filename
        for filename in required_buffer_files
        if not os.path.isfile(os.path.join(load_path, filename))
    ]
    if missing_files:
        raise FileNotFoundError(
            "Replay-buffer directory is incomplete. Missing: "
            + ", ".join(missing_files)
            + f". Directory: {load_path}"
        )

    if int(cfg.actor.global_batch_size) <= 0:
        raise ValueError("actor.global_batch_size must be positive.")

    if int(cfg.actor.micro_batch_size) <= 0:
        raise ValueError("actor.micro_batch_size must be positive.")

    # ckpt_path is not automatically handled by OfflineRunner. Full training
    # resume should use runner.resume_dir.
    ckpt_path = cfg.runner.get("ckpt_path", None)
    if ckpt_path:
        raise ValueError(
            "runner.ckpt_path is not supported by this entrypoint. "
            "Use runner.resume_dir pointing to a complete "
            "global_step_<N> checkpoint directory."
        )


@hydra.main(
    version_base="1.1",
    config_path="config",
    config_name="rlt_stage2_offline_ac_mlp",
)
def main(cfg: DictConfig) -> None:
    """Launch pure offline RLT Stage 2 actor-critic training."""

    cfg = _prepare_offline_cfg(cfg)

    # validate_cfg initializes standard RLinf defaults and validates the FSDP
    # actor configuration and component placement.
    cfg = validate_cfg(cfg)
    _validate_offline_rlt_cfg(cfg)

    print("Resolved offline RLT Stage 2 configuration:")
    print(
        json.dumps(
            OmegaConf.to_container(cfg, resolve=True),
            indent=2,
            ensure_ascii=False,
        )
    )

    cluster = Cluster(
        cluster_cfg=cfg.cluster,
        distributed_log_dir=cfg.runner.per_worker_log_path,
    )

    component_placement = HybridComponentPlacement(cfg, cluster)
    actor_placement = component_placement.get_strategy("actor")

    # This worker subclasses the original online RLT worker and only adds
    # offline replay-buffer loading. The original online worker remains
    # untouched for later real-robot training.
    actor_group = OfflineRLTACFSDPPolicy.create_group(cfg).launch(
        cluster,
        name=cfg.actor.group_name,
        placement_strategy=actor_placement,
    )

    runner = OfflineRunner(
        cfg=cfg,
        actor=actor_group,
        env=None,
        rollout=None,
    )

    runner.init_workers()
    runner.run()


if __name__ == "__main__":
    main()