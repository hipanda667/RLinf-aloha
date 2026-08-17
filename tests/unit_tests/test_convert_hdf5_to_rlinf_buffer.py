import numpy as np
import torch

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
