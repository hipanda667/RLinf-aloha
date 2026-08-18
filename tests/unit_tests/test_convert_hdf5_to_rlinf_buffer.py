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

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from rlinf.data.storage.replay import (
    TrajectoryReplayBuffer,
    validate_replay_checkpoint,
)
from toolkits.replay_buffer import convert_hdf5_to_rlinf_buffer as converter


def test_pad_take_uses_field_specific_terminal_padding():
    values = np.asarray([[1.0], [2.0]], dtype=np.float32)

    repeated = converter._pad_take(values, 1, 4)
    zero_padded = converter._pad_take(values, 1, 4, pad_value=0.0)

    np.testing.assert_array_equal(repeated[:, 0], np.asarray([2.0, 2.0, 2.0, 2.0]))
    np.testing.assert_array_equal(zero_padded[:, 0], np.asarray([2.0, 0.0, 0.0, 0.0]))


def test_chunk16_terminal_labels_are_not_duplicated():
    transition_length = 18
    observation_length = transition_length + 1
    num_chunks = 2
    episode = converter.EpisodeArrays(
        states=np.zeros((observation_length, 14), dtype=np.float32),
        actions=np.arange(transition_length * 14, dtype=np.float32).reshape(
            transition_length, 14
        ),
        rewards=np.concatenate(
            [
                np.zeros(transition_length - 1, dtype=np.float32),
                np.ones(1, dtype=np.float32),
            ]
        ),
        human_flags=np.asarray([False] * 16 + [True, True]),
        human_actions=None,
        main_images=np.zeros((observation_length, 2, 2, 3), dtype=np.uint8),
        wrist_images=None,
        extra_view_images=None,
    )
    curr_features = {
        "z_rl": torch.zeros(num_chunks, 2048),
        "proprio": torch.zeros(num_chunks, 14),
        "ref_chunk": torch.zeros(num_chunks, 16, 14),
    }
    next_features = {key: value.clone() for key, value in curr_features.items()}

    trajectory = converter.build_chunked_trajectory(
        episode=episode,
        curr_features=curr_features,
        next_features=next_features,
        starts=np.asarray([0, 16], dtype=np.int64),
        model_weights_id="stage1-test",
    )

    assert trajectory.actions.shape == (2, 1, 16 * 14)
    assert trajectory.rewards.shape == (2, 1, 16)
    assert trajectory.intervene_flags.shape == (2, 1, 16 * 14)
    assert trajectory.rewards.sum().item() == 1.0
    assert trajectory.intervene_flags.sum().item() == 2 * 14
    assert trajectory.terminations.sum().item() == 1
    assert trajectory.dones.sum().item() == 1
    assert trajectory.truncations.sum().item() == 0
    assert trajectory.terminations[-1, 0, 1]
    assert not trajectory.terminations[-1, 0, 2:].any()
    assert not trajectory.intervene_flags[-1, 0].reshape(16, 14)[2:].any()


def test_exact_chunk_boundary_is_terminal():
    starts = converter._transition_starts(16)

    np.testing.assert_array_equal(starts, np.asarray([0], dtype=np.int64))
    assert starts[-1] + converter.ACTION_CHUNK >= 16


class _FakeH5File(dict):
    """Small context-manager mapping used to exercise HDF5 alignment logic."""

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


def _fake_hdf5_arrays(*, observations: int, actions: int) -> dict[str, np.ndarray]:
    image_shape = (observations, 2, 3, 3)
    return {
        "observations/qpos": np.zeros((observations, 14), dtype=np.float32),
        "action": np.arange(actions * 14, dtype=np.float32).reshape(actions, 14),
        "reward": np.arange(actions, dtype=np.float32),
        "observations/images/cam_high": np.zeros(image_shape, dtype=np.uint8),
        "observations/images/cam_left_wrist": np.ones(image_shape, dtype=np.uint8),
        "observations/images/cam_right_wrist": np.full(image_shape, 2, dtype=np.uint8),
    }


@pytest.mark.parametrize("action_length", [4, 5])
def test_hdf5_loader_preserves_all_t_minus_one_transitions(monkeypatch, action_length):
    arrays = _fake_hdf5_arrays(observations=5, actions=action_length)
    fake_h5py = SimpleNamespace(File=lambda *_args, **_kwargs: _FakeH5File(arrays))
    monkeypatch.setattr(converter, "_require_h5py", lambda: fake_h5py)

    episode = converter.load_hdf5_episode("episode_0.hdf5")

    assert episode.states.shape == (5, 14)
    assert episode.actions.shape == (4, 14)
    assert episode.rewards.shape == (4,)
    assert episode.wrist_images is not None
    assert episode.extra_view_images is not None
    np.testing.assert_array_equal(
        episode.actions,
        arrays["action"][:4],
    )


def test_hdf5_loader_requires_both_wrist_cameras(monkeypatch):
    arrays = _fake_hdf5_arrays(observations=5, actions=4)
    del arrays["observations/images/cam_right_wrist"]
    fake_h5py = SimpleNamespace(File=lambda *_args, **_kwargs: _FakeH5File(arrays))
    monkeypatch.setattr(converter, "_require_h5py", lambda: fake_h5py)

    with pytest.raises(KeyError, match="cam_right_wrist"):
        converter.load_hdf5_episode("episode_0.hdf5")


def test_converted_trajectory_round_trips_through_current_replay_api(tmp_path):
    transition_length = 18
    episode = converter.EpisodeArrays(
        states=np.zeros((transition_length + 1, 14), dtype=np.float32),
        actions=np.zeros((transition_length, 14), dtype=np.float32),
        rewards=np.zeros((transition_length,), dtype=np.float32),
        human_flags=np.zeros((transition_length,), dtype=bool),
        human_actions=None,
        main_images=np.zeros((transition_length + 1, 2, 2, 3), dtype=np.uint8),
        wrist_images=np.zeros((transition_length + 1, 2, 2, 3), dtype=np.uint8),
        extra_view_images=np.zeros((transition_length + 1, 2, 2, 3), dtype=np.uint8),
    )
    starts = converter._transition_starts(transition_length)
    features = {
        "z_rl": torch.zeros(len(starts), 2048),
        "proprio": torch.zeros(len(starts), 14),
        "ref_chunk": torch.zeros(len(starts), 16, 14),
    }
    trajectory = converter.build_chunked_trajectory(
        episode=episode,
        curr_features=features,
        next_features={key: value.clone() for key, value in features.items()},
        starts=starts,
        model_weights_id="stage1-roundtrip",
    )
    checkpoint = tmp_path / "replay"

    writer = TrajectoryReplayBuffer(
        seed=1234,
        enable_cache=True,
        cache_size=2,
        sample_window_size=2,
        auto_save=False,
        trajectory_format="pt",
    )
    try:
        writer.add_trajectories([trajectory])
        writer.save_checkpoint(str(checkpoint))
    finally:
        writer.close()

    info = validate_replay_checkpoint(
        checkpoint,
        min_sample_count=2,
        world_size=1,
    )
    assert info.num_trajectories == 1
    assert info.total_samples == 2

    reader = TrajectoryReplayBuffer(
        seed=1234,
        enable_cache=True,
        cache_size=2,
        sample_window_size=2,
        auto_save=False,
        trajectory_format="pt",
    )
    try:
        reader.load_checkpoint(str(checkpoint))
        batch = reader.sample(num_chunks=2)
    finally:
        reader.close()

    assert batch["actions"].shape == (2, 16 * 14)
    assert batch["curr_obs"]["z_rl"].shape == (2, 2048)
    assert batch["curr_obs"]["proprio"].shape == (2, 14)
    assert batch["curr_obs"]["ref_chunk"].shape == (2, 16, 14)
