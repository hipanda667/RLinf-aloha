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

"""Pure-offline worker for RLT Stage-2 actor-critic training."""

from rlinf.data.storage.replay.validation import validate_replay_checkpoint
from rlinf.workers.actor.fsdp_rlt_ac_policy_worker import RLTACFSDPPolicy


class OfflineRLTACFSDPPolicy(RLTACFSDPPolicy):
    """Train the upstream RLT actor solely from a saved replay checkpoint."""

    def setup_sac_components(self) -> None:
        """Initialize SAC state and load this rank's validated replay shard."""
        if self.use_rlt_schedule:
            raise ValueError(
                "Offline RLT does not ingest rollout transitions. Set "
                "algorithm.rlt_schedule.enable=false."
            )

        super().setup_sac_components()
        replay_cfg = self.cfg.algorithm.replay_buffer
        load_path = replay_cfg.get("load_path")
        if not load_path:
            raise ValueError("Offline RLT requires algorithm.replay_buffer.load_path.")

        min_sample_count = int(
            replay_cfg.get(
                "min_sample_count",
                self.cfg.actor.global_batch_size,
            )
        )
        checkpoint_info = validate_replay_checkpoint(
            load_path,
            min_sample_count=min_sample_count,
            world_size=self._world_size,
        )
        self.replay_buffer.load_checkpoint(
            str(checkpoint_info.path),
            is_distributed=self._world_size > 1,
            local_rank=self._rank,
            world_size=self._world_size,
        )

        stats = self.replay_buffer.get_stats()
        min_trajectories = int(replay_cfg.get("min_buffer_size", 1))
        if stats["num_trajectories"] < min_trajectories:
            raise RuntimeError(
                f"Replay shard for actor rank {self._rank} has "
                f"{stats['num_trajectories']} trajectories, but "
                f"min_buffer_size={min_trajectories}."
            )

        per_rank_batch_size = int(self.cfg.actor.global_batch_size) // int(
            self._world_size
        )
        if stats["total_samples"] < per_rank_batch_size:
            raise RuntimeError(
                f"Replay shard for actor rank {self._rank} has "
                f"{stats['total_samples']} samples, but a local batch requires "
                f"{per_rank_batch_size}."
            )

        self.log_info(
            "Loaded offline RLT replay shard "
            f"rank={self._rank}/{self._world_size} path={checkpoint_info.path} "
            f"stats={stats}"
        )
