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

"""Fail-closed dual-arm ALOHA Gym environment.

The initial implementation is intentionally hardware-independent. A real ROS
backend must be injected by the ALOHA client after no-motion validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np

from .aloha_contract import (
    ACTION_DIM,
    CAMERA_NAMES,
    TASK_DESCRIPTION,
    make_raw_observation,
    require_aloha_vector,
)
from .aloha_hardware import AlohaHardware, DummyAlohaHardware
from .aloha_sandwich_reward import TerminalReason, outcome_for
from .aloha_teleop import ControlMode, select_executed_action


@dataclass
class AlohaConfig:
    """Configuration with safe defaults that cannot command a real robot."""

    is_dummy: bool = True
    send_actions: bool = False
    image_height: int = 480
    image_width: int = 640
    step_frequency: float = 25.0
    max_num_steps: int = 1000
    gripper_position_min: float = 0.0
    gripper_position_max: float = 0.08
    enable_human_in_loop: bool = False
    manual_episode_control_only: bool = False


class AlohaEnv(gym.Env):
    """Raw ALOHA environment with explicit actual-action bookkeeping."""

    metadata = {"render_modes": []}
    supports_relative_frame = False
    supports_leader_follower_keyboard_intervention = False

    def __init__(
        self,
        config: AlohaConfig,
        worker_info: Any = None,
        hardware_info: Any = None,
        env_idx: int = 0,
        *,
        hardware: AlohaHardware | None = None,
    ) -> None:
        del worker_info, hardware_info
        self.config = config
        self.env_idx = env_idx
        self.control_mode = ControlMode.MODEL
        self._num_steps = 0
        self._terminal_reason = TerminalReason.NONE

        if config.is_dummy:
            if hardware is not None:
                raise ValueError("Do not inject real hardware when is_dummy=True.")
            self.hardware: AlohaHardware = DummyAlohaHardware(
                config.image_height,
                config.image_width,
            )
        else:
            if hardware is None:
                raise RuntimeError(
                    "Real ALOHA hardware is not configured. Inject the validated "
                    "ALOHA client/ROS backend explicitly; this environment never "
                    "auto-connects to ROS."
                )
            if config.send_actions:
                raise RuntimeError(
                    "Real ALOHA action sending remains locked until joint limits, "
                    "single-step delta limits, watchdog timeout, and emergency-stop "
                    "handling are implemented and validated."
                )
            self.hardware = hardware

        self._init_spaces()

    @property
    def task_description(self) -> str:
        """Return the language instruction paired with observations."""
        return TASK_DESCRIPTION

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[dict, dict]:
        """Reset episode bookkeeping without commanding robot motion."""
        super().reset(seed=seed)
        del options
        self._num_steps = 0
        self._terminal_reason = TerminalReason.NONE
        self.control_mode = ControlMode.MODEL
        return self._get_observation(), {}

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, bool, dict]:
        """Select and optionally execute one validated absolute ALOHA action."""
        policy_action = require_aloha_vector(action, name="action")
        human_action = (
            self.hardware.read_human_action()
            if self.config.enable_human_in_loop
            and self.control_mode == ControlMode.TELEOP
            else None
        )
        decision = select_executed_action(
            policy_action,
            policy_source="policy",
            human_action=human_action,
        )

        if self.config.send_actions:
            self.hardware.send_action(decision.action)

        self._num_steps += 1
        if (
            self._terminal_reason == TerminalReason.NONE
            and self._num_steps >= self.config.max_num_steps
        ):
            self._terminal_reason = TerminalReason.TIMEOUT
        outcome = outcome_for(self._terminal_reason)

        info: dict[str, Any] = {
            "action_source": decision.source,
            "intervene_flag": decision.intervene_flag,
            "success": outcome.success,
            "terminal_reason": outcome.terminal_reason.value,
            "record_transition": bool(self.config.send_actions),
        }
        if decision.intervene_flag and self.config.send_actions:
            info["intervene_action"] = decision.action.copy()

        return (
            self._get_observation(),
            outcome.reward,
            outcome.terminated,
            outcome.truncated,
            info,
        )

    def request_terminal(self, reason: TerminalReason | str) -> None:
        """Set an explicit terminal label to be emitted by the next step."""
        self._terminal_reason = TerminalReason(reason)

    def set_control_mode(self, mode: ControlMode) -> None:
        """Set whether policy or human input may supply the next action."""
        self.control_mode = ControlMode(mode)

    def close(self) -> None:
        """Release resources owned by the injected hardware backend."""
        self.hardware.close()

    def _get_observation(self) -> dict:
        sample = self.hardware.read_observation()
        return make_raw_observation(sample.qpos, sample.frames)

    def _init_spaces(self) -> None:
        action_low = np.full(ACTION_DIM, -np.inf, dtype=np.float32)
        action_high = np.full(ACTION_DIM, np.inf, dtype=np.float32)
        action_low[[6, 13]] = self.config.gripper_position_min
        action_high[[6, 13]] = self.config.gripper_position_max
        self.action_space = gym.spaces.Box(
            low=action_low,
            high=action_high,
            dtype=np.float32,
        )

        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Dict(
                    {
                        "qpos": gym.spaces.Box(
                            -np.inf,
                            np.inf,
                            shape=(ACTION_DIM,),
                            dtype=np.float32,
                        )
                    }
                ),
                "frames": gym.spaces.Dict(
                    {
                        name: gym.spaces.Box(
                            0,
                            255,
                            shape=(
                                self.config.image_height,
                                self.config.image_width,
                                3,
                            ),
                            dtype=np.uint8,
                        )
                        for name in CAMERA_NAMES
                    }
                ),
            }
        )
