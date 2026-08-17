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

"""Strict observation and action contracts for the dual-arm ALOHA setup."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

NUM_JOINTS_PER_ARM = 6
ACTION_DIM = 14

ACTION_NAMES = (
    "left_joint_1",
    "left_joint_2",
    "left_joint_3",
    "left_joint_4",
    "left_joint_5",
    "left_joint_6",
    "left_gripper",
    "right_joint_1",
    "right_joint_2",
    "right_joint_3",
    "right_joint_4",
    "right_joint_5",
    "right_joint_6",
    "right_gripper",
)

MAIN_CAMERA_NAME = "cam_high"
LEFT_WRIST_CAMERA_NAME = "cam_left_wrist"
RIGHT_WRIST_CAMERA_NAME = "cam_right_wrist"
CAMERA_NAMES = (
    MAIN_CAMERA_NAME,
    LEFT_WRIST_CAMERA_NAME,
    RIGHT_WRIST_CAMERA_NAME,
)

TASK_DESCRIPTION = "make a sandwich"


def require_aloha_vector(
    value: np.ndarray,
    *,
    name: str,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """Return one copied 14D ALOHA vector, rejecting ambiguous shapes."""
    vector = np.asarray(value)
    if vector.shape != (ACTION_DIM,):
        raise ValueError(f"{name} must have shape ({ACTION_DIM},), got {vector.shape}.")
    if not np.issubdtype(vector.dtype, np.number):
        raise TypeError(f"{name} must be numeric, got dtype={vector.dtype}.")
    if not np.isfinite(vector).all():
        raise ValueError(f"{name} contains NaN or infinity.")
    return vector.astype(dtype, copy=True)


def require_rgb_frame(value: np.ndarray, *, camera_name: str) -> np.ndarray:
    """Validate one unbatched RGB/HWC/uint8 camera frame."""
    frame = np.asarray(value)
    if frame.ndim != 3 or frame.shape[-1] != 3:
        raise ValueError(
            f"{camera_name} must have RGB HWC shape [H,W,3], got {frame.shape}."
        )
    if frame.dtype != np.uint8:
        raise TypeError(f"{camera_name} must have dtype=uint8, got {frame.dtype}.")
    return np.ascontiguousarray(frame)


def make_raw_observation(
    qpos: np.ndarray,
    frames: Mapping[str, np.ndarray],
) -> dict[str, dict[str, np.ndarray]]:
    """Build the raw Gym observation consumed by RealWorldEnv.

    Camera identity is explicit. No resize, crop, channel transpose, batching,
    normalization, or tensor conversion is performed here.
    """
    state = require_aloha_vector(qpos, name="qpos")
    missing = [name for name in CAMERA_NAMES if name not in frames]
    if missing:
        raise KeyError(f"Missing required ALOHA cameras: {missing}.")

    validated_frames = {
        name: require_rgb_frame(frames[name], camera_name=name) for name in CAMERA_NAMES
    }
    frame_shapes = {frame.shape for frame in validated_frames.values()}
    if len(frame_shapes) != 1:
        raise ValueError(
            "All ALOHA camera frames must share one HWC shape so wrist images "
            f"can be stacked; got {sorted(frame_shapes)}."
        )

    return {
        "state": {"qpos": state},
        "frames": validated_frames,
    }
