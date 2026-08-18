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

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from rlinf.workers.sft.fsdp_vla_sft_worker import _copy_openpi_norm_stats


def _config(norm_stats_path: Path, repo_id: str = "aloha_sandwich"):
    return OmegaConf.create(
        {
            "actor": {
                "model": {
                    "openpi_data": {
                        "repo_id": repo_id,
                        "norm_stats_path": str(norm_stats_path),
                    }
                }
            }
        }
    )


def test_copy_openpi_norm_stats_makes_checkpoint_self_contained(tmp_path):
    source = tmp_path / "dataset" / "norm_stats.json"
    source.parent.mkdir()
    source.write_text('{"norm": "aloha"}', encoding="utf-8")

    checkpoint = tmp_path / "checkpoint"
    destination = _copy_openpi_norm_stats(_config(source), str(checkpoint))

    assert destination == checkpoint / "aloha_sandwich" / "norm_stats.json"
    assert destination.read_bytes() == source.read_bytes()
    assert (
        _copy_openpi_norm_stats(_config(source.parent), str(checkpoint)) == destination
    )


def test_copy_openpi_norm_stats_rejects_mismatch_and_unsafe_repo_id(tmp_path):
    source = tmp_path / "norm_stats.json"
    source.write_text('{"version": 1}', encoding="utf-8")
    checkpoint = tmp_path / "checkpoint"
    destination = checkpoint / "aloha_sandwich" / "norm_stats.json"
    destination.parent.mkdir(parents=True)
    destination.write_text('{"version": 2}', encoding="utf-8")

    with pytest.raises(ValueError, match="different norm stats"):
        _copy_openpi_norm_stats(_config(source), str(checkpoint))
    with pytest.raises(ValueError, match="Unsafe OpenPI repo_id"):
        _copy_openpi_norm_stats(_config(source, "../escape"), str(checkpoint))
