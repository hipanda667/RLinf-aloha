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

from argparse import Namespace

import numpy as np
import pytest
import torch

from examples.serving.scripts.serve_pi05_aloha import (
    RLinfOpenPiPolicy,
    _build_model_cfg,
    _normalise_observation,
    _random_aloha_observation,
)


def test_aloha_observation_preserves_all_three_cameras() -> None:
    observation = _random_aloha_observation("make a sandwich")

    normalized = _normalise_observation(observation, "unused")

    assert normalized["main_images"].shape == (1, 224, 224, 3)
    assert normalized["wrist_images"].shape == (1, 2, 224, 224, 3)
    assert normalized["states"].shape == (1, 14)
    assert normalized["task_descriptions"] == ["make a sandwich"]


def test_aloha_observation_rejects_missing_camera() -> None:
    observation = _random_aloha_observation("make a sandwich")
    del observation["images"]["cam_right_wrist"]

    with pytest.raises(KeyError, match="cam_right_wrist"):
        _normalise_observation(observation, "unused")


class _FakePolicyModel:
    def __init__(
        self,
        actions: np.ndarray,
        features: dict[str, np.ndarray] | None = None,
    ) -> None:
        self.actions = actions
        self.features = features or {
            "z_rl": np.zeros((1, 2048), dtype=np.float32),
            "proprio": np.zeros((1, 14), dtype=np.float32),
            "ref_chunk": np.zeros((1, 16, 14), dtype=np.float32),
        }

    def predict_action_batch(self, **_kwargs):
        return self.actions, {}

    def extract_rlt_obs(self, _env_obs):
        return self.features


def _policy_with_actions(
    actions: np.ndarray,
    features: dict[str, np.ndarray] | None = None,
) -> RLinfOpenPiPolicy:
    policy = object.__new__(RLinfOpenPiPolicy)
    policy._model = _FakePolicyModel(actions, features)
    policy._default_prompt = "make a sandwich"
    policy._action_horizon = 16
    policy._action_dim = 14
    return policy


def test_policy_server_accepts_finite_16_step_actions() -> None:
    policy = _policy_with_actions(np.zeros((1, 16, 14), dtype=np.float32))

    result = policy.infer(_random_aloha_observation("make a sandwich"))

    assert result["actions"].shape == (16, 14)
    assert np.isfinite(result["actions"]).all()


def test_policy_server_extracts_finite_stage2_features() -> None:
    policy = _policy_with_actions(np.zeros((1, 16, 14), dtype=np.float32))

    features = policy.extract_stage2_features(
        _random_aloha_observation("make a sandwich")
    )

    assert features["z_rl"].shape == (1, 2048)
    assert features["proprio"].shape == (1, 14)
    assert features["ref_chunk"].shape == (1, 16, 14)
    assert all(np.isfinite(value).all() for value in features.values())


@pytest.mark.parametrize(
    "features, message",
    [
        (
            {
                "z_rl": np.zeros((1, 1024), dtype=np.float32),
                "proprio": np.zeros((1, 14), dtype=np.float32),
                "ref_chunk": np.zeros((1, 16, 14), dtype=np.float32),
            },
            "z_rl has shape",
        ),
        (
            {
                "z_rl": np.zeros((1, 2048), dtype=np.float32),
                "proprio": np.zeros((1, 14), dtype=np.float32),
            },
            "missing 'ref_chunk'",
        ),
        (
            {
                "z_rl": np.full((1, 2048), np.nan, dtype=np.float32),
                "proprio": np.zeros((1, 14), dtype=np.float32),
                "ref_chunk": np.zeros((1, 16, 14), dtype=np.float32),
            },
            "non-finite",
        ),
    ],
)
def test_policy_server_rejects_invalid_stage2_features(
    features: dict[str, np.ndarray],
    message: str,
) -> None:
    policy = _policy_with_actions(
        np.zeros((1, 16, 14), dtype=np.float32),
        features,
    )

    with pytest.raises(RuntimeError, match=message):
        policy.extract_stage2_features(_random_aloha_observation("make a sandwich"))


@pytest.mark.parametrize(
    "actions, message",
    [
        (np.zeros((1, 8, 14), dtype=np.float32), "expected"),
        (np.full((1, 16, 14), np.nan, dtype=np.float32), "non-finite"),
    ],
)
def test_policy_server_rejects_invalid_actions(
    actions: np.ndarray, message: str
) -> None:
    policy = _policy_with_actions(actions)

    with pytest.raises(RuntimeError, match=message):
        policy.infer(_random_aloha_observation("make a sandwich"))


def test_legacy_checkpoint_format_is_rejected(tmp_path) -> None:
    config_path = tmp_path / "legacy.yaml"
    config_path.write_text(
        """
server:
  strict_load: true
checkpoint:
  format: legacy
  dir: /placeholder
  repo_id: aloha_sandwich
  default_prompt: make a sandwich
"""
    )
    args = Namespace(
        config=str(config_path),
        checkpoint_dir=None,
        checkpoint_format=None,
        repo_id=None,
        default_prompt=None,
    )

    with pytest.raises(RuntimeError, match="legacy/main-3eeb9265"):
        _build_model_cfg(args)


def test_openpi_rlinf_strict_checkpoint_load(tmp_path) -> None:
    from rlinf.models.embodiment.openpi_rlinf.utils.rlt_utils import (
        load_full_wrapper_weights,
    )

    wrapper = torch.nn.Linear(2, 2)
    checkpoint_path = tmp_path / "full_weights.pt"
    torch.save(wrapper.state_dict(), checkpoint_path)
    load_full_wrapper_weights(
        wrapper,
        checkpoint_path,
        expect_rlt=False,
        strict=True,
    )

    incomplete = wrapper.state_dict()
    incomplete.pop("bias")
    torch.save(incomplete, checkpoint_path)
    with pytest.raises(RuntimeError, match="Missing key"):
        load_full_wrapper_weights(
            wrapper,
            checkpoint_path,
            expect_rlt=False,
            strict=True,
        )
