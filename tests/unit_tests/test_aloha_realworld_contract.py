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

import numpy as np
import pytest
import torch

from rlinf.envs.realworld.aloha.aloha_contract import (
    ACTION_NAMES,
    CAMERA_NAMES,
    make_raw_observation,
    require_aloha_vector,
)
from rlinf.envs.realworld.aloha.aloha_env import AlohaConfig, AlohaEnv
from rlinf.envs.realworld.aloha.aloha_sandwich_reward import (
    TerminalReason,
    outcome_for,
)
from rlinf.envs.realworld.aloha.aloha_teleop import select_executed_action
from rlinf.envs.realworld.realworld_env import RealWorldEnv


def _frames(height: int = 8, width: int = 12) -> dict[str, np.ndarray]:
    return {
        name: np.full((height, width, 3), fill, dtype=np.uint8)
        for name, fill in zip(CAMERA_NAMES, (16, 96, 176), strict=True)
    }


def test_action_contract_is_left_first_and_strictly_14d():
    assert ACTION_NAMES[0] == "left_joint_1"
    assert ACTION_NAMES[6] == "left_gripper"
    assert ACTION_NAMES[7] == "right_joint_1"
    assert ACTION_NAMES[13] == "right_gripper"

    vector = require_aloha_vector(np.arange(14), name="action")
    assert vector.shape == (14,)
    assert vector.dtype == np.float32

    with pytest.raises(ValueError, match="shape"):
        require_aloha_vector(np.zeros((1, 14)), name="action")


def test_raw_observation_requires_three_rgb_hwc_uint8_frames():
    obs = make_raw_observation(np.zeros(14), _frames())
    assert tuple(obs["frames"]) == CAMERA_NAMES
    assert obs["state"]["qpos"].shape == (14,)

    missing = _frames()
    missing.pop("cam_left_wrist")
    with pytest.raises(KeyError, match="cam_left_wrist"):
        make_raw_observation(np.zeros(14), missing)

    wrong_dtype = _frames()
    wrong_dtype["cam_high"] = wrong_dtype["cam_high"].astype(np.float32)
    with pytest.raises(TypeError, match="uint8"):
        make_raw_observation(np.zeros(14), wrong_dtype)


def test_realworld_wrapper_keeps_explicit_left_right_wrist_order():
    env = RealWorldEnv.__new__(RealWorldEnv)
    env.main_image_key = "cam_high"
    env.wrist_image_keys = ("cam_left_wrist", "cam_right_wrist")
    env.task_descriptions = ["make a sandwich"]

    raw_obs = {
        "state": {"qpos": np.arange(14, dtype=np.float32)[None, :]},
        "frames": {
            name: frame[None, ...] for name, frame in _frames().items()
        },
    }
    obs = env._wrap_obs(raw_obs)

    assert tuple(obs) == (
        "states",
        "main_images",
        "wrist_images",
        "extra_view_images",
        "task_descriptions",
    )
    assert obs["states"].shape == (1, 14)
    assert obs["main_images"].shape == (1, 8, 12, 3)
    assert obs["wrist_images"].shape == (1, 2, 8, 12, 3)
    assert torch.all(obs["wrist_images"][:, 0] == 96)
    assert torch.all(obs["wrist_images"][:, 1] == 176)
    assert obs["extra_view_images"] is None


def test_dummy_env_is_observation_only_by_default():
    env = AlohaEnv(AlohaConfig(image_height=8, image_width=12))
    obs, info = env.reset()
    assert info == {}
    assert env.observation_space.contains(obs)

    next_obs, reward, terminated, truncated, info = env.step(
        np.zeros(14, dtype=np.float32)
    )
    assert env.observation_space.contains(next_obs)
    assert reward == 0.0
    assert terminated is False
    assert truncated is False
    assert info["action_source"] == "policy"
    assert info["intervene_flag"] is False
    assert info["record_transition"] is False


def test_dummy_action_send_is_explicit_and_updates_dummy_state():
    env = AlohaEnv(
        AlohaConfig(
            is_dummy=True,
            send_actions=True,
            image_height=8,
            image_width=12,
        )
    )
    env.reset()
    action = np.linspace(-0.5, 0.5, 14, dtype=np.float32)
    action[[6, 13]] = (0.02, 0.06)
    obs, *_ = env.step(action)
    np.testing.assert_allclose(obs["state"]["qpos"], action)


def test_human_action_has_priority_and_is_independently_flagged():
    policy_action = np.zeros(14, dtype=np.float32)
    human_action = np.ones(14, dtype=np.float32)
    decision = select_executed_action(
        policy_action,
        policy_source="actor",
        human_action=human_action,
    )
    assert decision.source == "human"
    assert decision.intervene_flag is True
    np.testing.assert_array_equal(decision.action, human_action)


@pytest.mark.parametrize(
    ("reason", "expected"),
    [
        (TerminalReason.NONE, (0.0, False, False, False)),
        (TerminalReason.SUCCESS, (1.0, True, False, True)),
        (TerminalReason.FAILURE, (0.0, True, False, False)),
        (TerminalReason.TIMEOUT, (0.0, False, True, False)),
    ],
)
def test_terminal_reason_is_independent_from_intervention(reason, expected):
    result = outcome_for(reason)
    assert (
        result.reward,
        result.terminated,
        result.truncated,
        result.success,
    ) == expected
