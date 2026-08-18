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

import json
from pathlib import Path

import pytest

from rlinf.data.storage.replay.validation import validate_replay_checkpoint


def _write_replay_checkpoint(
    path: Path,
    *,
    sample_counts: tuple[int, ...] = (16, 16),
) -> None:
    trajectory_index = {}
    trajectory_ids = list(range(len(sample_counts)))
    for trajectory_id, num_samples in enumerate(sample_counts):
        trajectory_index[str(trajectory_id)] = {
            "trajectory_id": trajectory_id,
            "num_samples": num_samples,
            "max_episode_length": num_samples,
            "shape": [num_samples, 1],
            "model_weights_id": "stage1",
        }
        (path / f"trajectory_{trajectory_id}_stage1.pt").write_bytes(b"test")

    (path / "metadata.json").write_text(
        json.dumps(
            {
                "trajectory_format": "pt",
                "size": len(sample_counts),
                "total_samples": sum(sample_counts),
                "trajectory_counter": len(sample_counts),
                "seed": 1234,
            }
        ),
        encoding="utf-8",
    )
    (path / "trajectory_index.json").write_text(
        json.dumps(
            {
                "trajectory_index": trajectory_index,
                "trajectory_id_list": trajectory_ids,
            }
        ),
        encoding="utf-8",
    )


def test_validate_replay_checkpoint_accepts_rank_shardable_buffer(tmp_path):
    _write_replay_checkpoint(tmp_path)

    info = validate_replay_checkpoint(
        tmp_path,
        min_sample_count=32,
        world_size=2,
    )

    assert info.path == tmp_path.resolve()
    assert info.trajectory_format == "pt"
    assert info.num_trajectories == 2
    assert info.total_samples == 32


def test_validate_replay_checkpoint_requires_metadata_and_index(tmp_path):
    with pytest.raises(FileNotFoundError, match="metadata.json"):
        validate_replay_checkpoint(
            tmp_path,
            min_sample_count=1,
        )


def test_validate_replay_checkpoint_rejects_inconsistent_sample_count(tmp_path):
    _write_replay_checkpoint(tmp_path)
    metadata_path = tmp_path / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["total_samples"] = 31
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="sample count is inconsistent"):
        validate_replay_checkpoint(
            tmp_path,
            min_sample_count=1,
        )


def test_validate_replay_checkpoint_requires_nonempty_shard_per_rank(tmp_path):
    _write_replay_checkpoint(tmp_path, sample_counts=(32,))

    with pytest.raises(ValueError, match="every rank"):
        validate_replay_checkpoint(
            tmp_path,
            min_sample_count=32,
            world_size=2,
        )
