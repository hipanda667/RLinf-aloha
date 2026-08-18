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

"""Validation helpers for replay-buffer checkpoints used by offline RL."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ReplayCheckpointInfo:
    """Validated summary of a replay-buffer checkpoint."""

    path: Path
    trajectory_format: str
    num_trajectories: int
    total_samples: int


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as file:
            value = json.load(file)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Replay metadata is not valid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Replay metadata must contain a JSON object: {path}")
    return value


def _nonnegative_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a non-negative integer, got {value!r}.")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{field} must be a non-negative integer, got {value!r}."
        ) from exc
    if result < 0 or result != value:
        raise ValueError(f"{field} must be a non-negative integer, got {value!r}.")
    return result


def validate_replay_checkpoint(
    load_path: str | Path,
    *,
    min_sample_count: int,
    world_size: int = 1,
) -> ReplayCheckpointInfo:
    """Validate a replay checkpoint before distributed offline training.

    Args:
        load_path: Directory written by TrajectoryReplayBuffer.save_checkpoint.
        min_sample_count: Minimum number of indexed transitions required.
        world_size: Number of actor ranks that will shard the trajectory list.

    Returns:
        A validated checkpoint summary.

    Raises:
        FileNotFoundError: If metadata, index, or trajectory files are missing.
        ValueError: If metadata is malformed or inconsistent.
    """
    path = Path(load_path).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Replay-buffer directory does not exist: {path}")

    min_sample_count = _nonnegative_int(min_sample_count, field="min_sample_count")
    world_size = _nonnegative_int(world_size, field="world_size")
    if world_size < 1:
        raise ValueError(f"world_size must be at least 1, got {world_size}.")

    metadata_path = path / "metadata.json"
    index_path = path / "trajectory_index.json"
    missing = [
        str(required)
        for required in (metadata_path, index_path)
        if not required.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Replay-buffer checkpoint is incomplete; missing: " + ", ".join(missing)
        )

    metadata = _load_json_object(metadata_path)
    index_data = _load_json_object(index_path)
    trajectory_format = str(metadata.get("trajectory_format", "pt"))
    if trajectory_format not in {"pt", "pkl"}:
        raise ValueError(
            "metadata.json trajectory_format must be 'pt' or 'pkl', "
            f"got {trajectory_format!r}."
        )

    trajectory_ids = index_data.get("trajectory_id_list")
    trajectory_index = index_data.get("trajectory_index")
    if not isinstance(trajectory_ids, list):
        raise ValueError(
            "trajectory_index.json trajectory_id_list must be a JSON list."
        )
    if not isinstance(trajectory_index, dict):
        raise ValueError(
            "trajectory_index.json trajectory_index must be a JSON object."
        )

    normalized_ids: list[int] = []
    indexed_samples = 0
    for raw_id in trajectory_ids:
        trajectory_id = _nonnegative_int(
            raw_id, field="trajectory_index.json trajectory id"
        )
        if trajectory_id in normalized_ids:
            raise ValueError(
                f"Duplicate trajectory id in replay index: {trajectory_id}."
            )
        normalized_ids.append(trajectory_id)

        entry = trajectory_index.get(str(trajectory_id))
        if not isinstance(entry, dict):
            raise ValueError(
                f"Replay index is missing trajectory metadata for id {trajectory_id}."
            )
        num_samples = _nonnegative_int(
            entry.get("num_samples"),
            field=f"trajectory {trajectory_id} num_samples",
        )
        indexed_samples += num_samples

        model_weights_id = entry.get("model_weights_id")
        if model_weights_id is None:
            raise ValueError(
                f"Replay trajectory {trajectory_id} has no model_weights_id."
            )
        trajectory_path = path / (
            f"trajectory_{trajectory_id}_{model_weights_id}.{trajectory_format}"
        )
        if not trajectory_path.is_file():
            raise FileNotFoundError(
                f"Replay trajectory file does not exist: {trajectory_path}"
            )

    metadata_size = _nonnegative_int(metadata.get("size"), field="metadata.json size")
    metadata_samples = _nonnegative_int(
        metadata.get("total_samples"), field="metadata.json total_samples"
    )
    if metadata_size != len(normalized_ids):
        raise ValueError(
            "Replay trajectory count is inconsistent: "
            f"metadata.json size={metadata_size}, "
            f"trajectory_index.json count={len(normalized_ids)}."
        )
    if metadata_samples != indexed_samples:
        raise ValueError(
            "Replay sample count is inconsistent: "
            f"metadata.json total_samples={metadata_samples}, "
            f"trajectory_index.json total={indexed_samples}."
        )
    if metadata_samples < min_sample_count:
        raise ValueError(
            f"Replay checkpoint has {metadata_samples} samples, "
            f"but min_sample_count={min_sample_count}."
        )
    if len(normalized_ids) < world_size:
        raise ValueError(
            f"Replay checkpoint has {len(normalized_ids)} trajectories for "
            f"{world_size} actor ranks; every rank must receive a non-empty shard."
        )

    return ReplayCheckpointInfo(
        path=path,
        trajectory_format=trajectory_format,
        num_trajectories=len(normalized_ids),
        total_samples=metadata_samples,
    )
