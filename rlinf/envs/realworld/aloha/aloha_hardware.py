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

"""Hardware boundary for ALOHA.

This module deliberately has no ROS imports. The real ROS implementation lives
on the ALOHA client and is injected only after observation-only tests pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .aloha_contract import CAMERA_NAMES, require_aloha_vector


@dataclass(frozen=True)
class AlohaHardwareObservation:
    qpos: np.ndarray
    frames: dict[str, np.ndarray]


class AlohaHardware(Protocol):
    """Minimal interface required by the hardware-independent Gym adapter."""

    def read_observation(self) -> AlohaHardwareObservation:
        """Read follower qpos and three RGB camera frames."""

    def send_action(self, action: np.ndarray) -> None:
        """Send one safe, absolute 14D follower target."""

    def read_human_action(self) -> np.ndarray | None:
        """Return one 14D human target while intervention is active."""

    def close(self) -> None:
        """Release hardware resources."""


class DummyAlohaHardware:
    """Deterministic no-motion backend for contract and wrapper tests."""

    def __init__(self, image_height: int, image_width: int) -> None:
        self._qpos = np.zeros(14, dtype=np.float32)
        # Distinct values make left/right camera swaps visible in tests.
        fill_values = (16, 96, 176)
        self._frames = {
            name: np.full(
                (image_height, image_width, 3),
                fill_value,
                dtype=np.uint8,
            )
            for name, fill_value in zip(CAMERA_NAMES, fill_values, strict=True)
        }
        self.last_action: np.ndarray | None = None

    def read_observation(self) -> AlohaHardwareObservation:
        return AlohaHardwareObservation(
            qpos=self._qpos.copy(),
            frames={name: frame.copy() for name, frame in self._frames.items()},
        )

    def send_action(self, action: np.ndarray) -> None:
        self.last_action = require_aloha_vector(action, name="dummy action")
        self._qpos = self.last_action.copy()

    def read_human_action(self) -> np.ndarray | None:
        return None

    def close(self) -> None:
        return None
