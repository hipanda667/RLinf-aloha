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

"""Pure action-source selection for ALOHA human-in-the-loop control."""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Literal

import numpy as np

from .aloha_contract import require_aloha_vector

PolicyActionSource = Literal["policy", "reference", "actor"]
ExecutedActionSource = Literal["policy", "reference", "actor", "human"]


class ControlMode(enum.IntEnum):
    MODEL = 0
    PAUSE = 1
    TELEOP = 2


@dataclass(frozen=True)
class ActionDecision:
    action: np.ndarray
    source: ExecutedActionSource
    intervene_flag: bool


def select_executed_action(
    policy_action: np.ndarray,
    *,
    policy_source: PolicyActionSource,
    human_action: np.ndarray | None,
) -> ActionDecision:
    """Apply ``human > actor/reference`` priority without touching hardware."""
    if policy_source not in ("policy", "reference", "actor"):
        raise ValueError(f"Unsupported policy action source: {policy_source!r}.")

    if human_action is not None:
        return ActionDecision(
            action=require_aloha_vector(human_action, name="human_action"),
            source="human",
            intervene_flag=True,
        )

    return ActionDecision(
        action=require_aloha_vector(policy_action, name="policy_action"),
        source=policy_source,
        intervene_flag=False,
    )
