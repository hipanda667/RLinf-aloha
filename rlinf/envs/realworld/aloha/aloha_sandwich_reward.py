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

"""Sparse terminal reward semantics matching the sandwich HDF5 data."""

from __future__ import annotations

import enum
from dataclasses import dataclass


class TerminalReason(str, enum.Enum):
    NONE = "none"
    SUCCESS = "success"
    FAILURE = "failure"
    UNRECOVERABLE_FAILURE = "unrecoverable_failure"
    USER_ABORT = "user_abort"
    TIMEOUT = "timeout"


@dataclass(frozen=True)
class SandwichStepOutcome:
    reward: float
    terminated: bool
    truncated: bool
    success: bool
    terminal_reason: TerminalReason


def outcome_for(reason: TerminalReason) -> SandwichStepOutcome:
    """Map one explicit terminal reason to Gym and replay-buffer fields."""
    if reason == TerminalReason.SUCCESS:
        return SandwichStepOutcome(1.0, True, False, True, reason)
    if reason == TerminalReason.TIMEOUT:
        return SandwichStepOutcome(0.0, False, True, False, reason)
    if reason in (
        TerminalReason.FAILURE,
        TerminalReason.UNRECOVERABLE_FAILURE,
        TerminalReason.USER_ABORT,
    ):
        return SandwichStepOutcome(0.0, True, False, False, reason)
    if reason == TerminalReason.NONE:
        return SandwichStepOutcome(0.0, False, False, False, reason)
    raise ValueError(f"Unsupported terminal reason: {reason!r}.")
