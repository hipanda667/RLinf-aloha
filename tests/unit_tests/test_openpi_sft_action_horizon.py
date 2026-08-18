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

import dataclasses
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest

from rlinf.data.datasets.openpi_rlinf import official_sft_data_loader


class _AttrDict(dict):
    """Dictionary exposing configuration keys as attributes."""

    __getattr__ = dict.__getitem__


@dataclasses.dataclass(frozen=True)
class _FakeModelConfig:
    action_horizon: int
    action_dim: int = 32


@dataclasses.dataclass(frozen=True)
class _FakeTrainConfig:
    model: _FakeModelConfig
    batch_size: int
    num_workers: int = 0
    seed: int = 0


class _FakeOpenPiLoader:
    """Small loader whose action shape follows the official model horizon."""

    def __init__(self, config: _FakeTrainConfig) -> None:
        self.config = config

    def __iter__(self):
        yield {
            "actions": np.zeros(
                (self.config.batch_size, self.config.model.action_horizon, 14),
                dtype=np.float32,
            )
        }

    def data_config(self) -> SimpleNamespace:
        return SimpleNamespace(action_horizon=self.config.model.action_horizon)


def _legacy_cfg(*, horizon: int = 16, output_chunk: int = 16) -> SimpleNamespace:
    model_cfg = SimpleNamespace(
        model_type="openpi",
        model_path="/placeholder/model",
        num_action_chunks=horizon,
        openpi=_AttrDict(
            config_name="pi05_aloha_robotwin",
            action_horizon=horizon,
            action_chunk=output_chunk,
        ),
        openpi_data=None,
    )
    return SimpleNamespace(
        actor=_AttrDict(model=model_cfg, micro_batch_size=2, seed=42),
        data=SimpleNamespace(),
    )


def _install_fake_openpi(monkeypatch: pytest.MonkeyPatch) -> None:
    openpi_module = types.ModuleType("openpi")
    openpi_module.__path__ = []
    training_module = types.ModuleType("openpi.training")
    training_module.__path__ = []
    loader_module = types.ModuleType("openpi.training.data_loader")
    loader_module.create_data_loader = lambda config, **_kwargs: _FakeOpenPiLoader(
        config
    )
    openpi_module.training = training_module
    training_module.data_loader = loader_module

    dataconfig_module = types.ModuleType("rlinf.models.embodiment.openpi.dataconfig")
    dataconfig_module.get_openpi_config = lambda _config_name, **kwargs: (
        _FakeTrainConfig(
            model=_FakeModelConfig(action_horizon=50),
            batch_size=kwargs["batch_size"],
        )
    )

    monkeypatch.setitem(sys.modules, "openpi", openpi_module)
    monkeypatch.setitem(sys.modules, "openpi.training", training_module)
    monkeypatch.setitem(sys.modules, "openpi.training.data_loader", loader_module)
    monkeypatch.setitem(
        sys.modules,
        "rlinf.models.embodiment.openpi.dataconfig",
        dataconfig_module,
    )


def test_legacy_aloha_dataset_model_and_output_horizons_are_16(monkeypatch):
    _install_fake_openpi(monkeypatch)
    monkeypatch.setattr(
        official_sft_data_loader,
        "resolve_lerobot_repo_id",
        lambda _paths: "aloha/sandwich",
    )
    cfg = _legacy_cfg()

    data_loader, data_config = (
        official_sft_data_loader.build_official_openpi_sft_dataloader(
            cfg,
            world_size=1,
            rank=0,
            data_paths="/placeholder/data",
        )
    )
    batch = next(iter(data_loader))

    assert data_config.action_horizon == 16
    assert data_loader.config.model.action_horizon == 16
    assert batch["actions"].shape == (2, 16, 14)
    assert cfg.actor.model.openpi.action_chunk == 16


def test_legacy_aloha_horizon_rejects_mismatched_output_chunk(monkeypatch):
    _install_fake_openpi(monkeypatch)
    monkeypatch.setattr(
        official_sft_data_loader,
        "resolve_lerobot_repo_id",
        lambda _paths: "aloha/sandwich",
    )

    with pytest.raises(ValueError, match="model and output chunk"):
        official_sft_data_loader.build_official_openpi_sft_dataloader(
            _legacy_cfg(output_chunk=8),
            world_size=1,
            rank=0,
            data_paths="/placeholder/data",
        )


@pytest.mark.parametrize(
    ("model_horizon", "model_action_dim", "message"),
    [
        (8, 32, "action horizon must match"),
        (16, 16, "model action dim must match"),
    ],
)
def test_openpi_rlinf_shape_validation_rejects_config_drift(
    model_horizon, model_action_dim, message
):
    model_cfg = SimpleNamespace(
        num_action_chunks=model_horizon,
        openpi=SimpleNamespace(
            config_name="pi05_aloha_robotwin_sandwich",
            model_action_dim=model_action_dim,
        ),
    )
    openpi_config = _FakeTrainConfig(
        model=_FakeModelConfig(action_horizon=16, action_dim=32),
        batch_size=1,
    )

    with pytest.raises(ValueError, match=message):
        official_sft_data_loader._validate_openpi_rlinf_model_shape(
            model_cfg, openpi_config
        )
