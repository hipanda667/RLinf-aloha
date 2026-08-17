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

"""Gym registration for the ALOHA sandwich task."""

from __future__ import annotations

from typing import Any, Mapping

import gymnasium as gym
from gymnasium.envs.registration import register

from rlinf.envs.realworld.aloha.aloha_env import AlohaConfig, AlohaEnv


def create_aloha_sandwich_env(
    override_cfg: dict[str, Any],
    worker_info: Any,
    hardware_info: Any,
    env_idx: int,
    env_cfg: Mapping[str, Any],
) -> gym.Env:
    """Build an ALOHA sandwich environment for RealWorldEnv."""
    del env_cfg
    return AlohaEnv(
        AlohaConfig(**override_cfg),
        worker_info=worker_info,
        hardware_info=hardware_info,
        env_idx=env_idx,
    )


register(
    id="AlohaSandwichEnv-v1",
    entry_point="rlinf.envs.realworld.aloha.tasks:create_aloha_sandwich_env",
)

__all__ = ["create_aloha_sandwich_env"]
