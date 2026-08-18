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

import torch

from rlinf.models.embodiment.mlp_policy.rlt_mlp_policy import RLTMLPPolicy


def _make_policy(squash_actions: bool) -> RLTMLPPolicy:
    policy = RLTMLPPolicy(
        z_dim=2048,
        proprio_dim=14,
        action_dim=14,
        num_action_chunks=16,
        ref_num_action_chunks=16,
        fixed_std=0.002,
        squash_actions=squash_actions,
    )
    with torch.no_grad():
        policy.actor_mean.weight.zero_()
        policy.actor_mean.bias.fill_(2.0)
    return policy


def _zero_obs(batch_size: int = 2) -> dict[str, torch.Tensor]:
    return {
        "z_rl": torch.zeros(batch_size, 1, 2048),
        "proprio": torch.zeros(batch_size, 1, 14),
        "ref_chunk": torch.zeros(batch_size, 1, 16, 14),
    }


def test_rlt_policy_action_squashing_is_configurable() -> None:
    obs = _zero_obs()
    with torch.no_grad():
        squashed, _, _ = _make_policy(True).sac_forward(obs, deterministic=True)
        unsquashed, _, _ = _make_policy(False).sac_forward(obs, deterministic=True)

    assert squashed.shape == (2, 224)
    assert unsquashed.shape == (2, 224)
    torch.testing.assert_close(
        squashed,
        torch.full_like(squashed, torch.tanh(torch.tensor(2.0))),
    )
    torch.testing.assert_close(unsquashed, torch.full_like(unsquashed, 2.0))
