"""Offline RLT actor-critic worker."""

from rlinf.workers.actor.rlt_ac_policy_worker import RLTACFSDPPolicy


class OfflineRLTACFSDPPolicy(RLTACFSDPPolicy):
    """RLT actor-critic worker trained from a saved replay buffer."""

    def setup_sac_components(self) -> None:
        super().setup_sac_components()

        load_path = self.cfg.algorithm.replay_buffer.get("load_path", None)
        if not load_path:
            raise ValueError(
                "Offline RLT training requires "
                "algorithm.replay_buffer.load_path."
            )

        self.replay_buffer.load_checkpoint(
            load_path,
            is_distributed=self._world_size > 1,
            local_rank=self._rank,
            world_size=self._world_size,
        )

        stats = self.replay_buffer.get_stats()
        min_buffer_size = int(
            self.cfg.algorithm.replay_buffer.min_buffer_size
        )

        if stats["total_samples"] < min_buffer_size:
            raise RuntimeError(
                f"Loaded replay buffer has only "
                f"{stats['total_samples']} samples, "
                f"but min_buffer_size={min_buffer_size}."
            )

        self.logger.info(
            "Loaded offline RLT replay buffer from %s: %s",
            load_path,
            stats,
        )