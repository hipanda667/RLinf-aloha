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

import re
from pathlib import Path

from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
STAGE1_PATH = (
    REPO_ROOT / "examples/sft/config/aloha_sandwich_rlt_stage1_sft_openpi_pi05.yaml"
)
ONLINE_PATH = (
    REPO_ROOT
    / "examples/embodiment/config/aloha_sandwich_rlt_stage2_online_ac_mlp.yaml"
)
OFFLINE_PATH = (
    REPO_ROOT
    / "examples/embodiment/config/aloha_sandwich_rlt_stage2_offline_ac_mlp.yaml"
)
ENV_PATH = REPO_ROOT / "examples/embodiment/config/env/realworld_aloha_sandwich.yaml"
SERVING_PATH = REPO_ROOT / "examples/serving/config/serve_pi05_aloha_sandwich.yaml"
LEGACY_RUNTIME_PATH = (
    REPO_ROOT / "examples/serving/config/legacy_pi05_aloha_runtime.yaml"
)
LEGACY_REQUIREMENTS_PATH = (
    REPO_ROOT / "requirements/legacy/aloha_pi05_legacy_runtime.txt"
)


def test_aloha_stage_contract_is_consistent_across_configs():
    stage1 = OmegaConf.load(STAGE1_PATH)
    online = OmegaConf.load(ONLINE_PATH)
    offline = OmegaConf.load(OFFLINE_PATH)
    serving = OmegaConf.load(SERVING_PATH)
    env = OmegaConf.load(ENV_PATH)

    stage1_model = stage1.actor.model
    feature_model = online.rollout.rlt_feature_model
    serving_model = serving.model
    online_actor = online.actor.model
    offline_actor = offline.actor.model

    assert stage1_model.model_type == feature_model.model_type == "openpi_rlinf"
    assert serving.checkpoint.format == "openpi_rlinf"
    assert serving.server.strict_load is True
    assert stage1_model.precision == "bf16"
    assert feature_model.precision == "bf16"
    assert serving_model.precision == "bf16"

    assert {
        int(stage1_model.num_action_chunks),
        int(stage1_model.openpi.action_horizon),
        int(stage1_model.openpi.action_chunk),
        int(feature_model.num_action_chunks),
        int(feature_model.openpi.action_horizon),
        int(feature_model.openpi.action_chunk),
        int(serving_model.num_action_chunks),
        int(serving_model.openpi.action_horizon),
        int(serving_model.openpi.action_chunk),
        int(online_actor.num_action_chunks),
        int(online_actor.ref_num_action_chunks),
        int(offline_actor.num_action_chunks),
        int(offline_actor.ref_num_action_chunks),
    } == {16}

    assert {
        int(stage1_model.action_dim),
        int(stage1_model.openpi.action_env_dim),
        int(feature_model.action_dim),
        int(feature_model.openpi.action_env_dim),
        int(serving_model.action_dim),
        int(serving_model.openpi.action_env_dim),
        int(online_actor.action_dim),
        int(online_actor.proprio_dim),
        int(offline_actor.action_dim),
        int(offline_actor.proprio_dim),
    } == {14}

    assert {
        stage1_model.openpi.config_name,
        feature_model.openpi.config_name,
        serving_model.config_name,
        serving_model.openpi.config_name,
    } == {"pi05_aloha_robotwin_sandwich"}
    assert {
        stage1_model.openpi_data.repo_id,
        feature_model.openpi_data.repo_id,
        serving.checkpoint.repo_id,
    } == {"aloha_sandwich"}
    assert {
        stage1_model.openpi_data.default_prompt,
        feature_model.openpi_data.default_prompt,
        serving.checkpoint.default_prompt,
    } == {"make a sandwich"}

    assert online_actor.squash_actions is False
    assert offline_actor.squash_actions is False
    assert offline.algorithm.rlt_schedule.enable is False
    assert offline.runner.val_check_interval == -1
    assert offline.runner.only_eval is False
    offline_text = OFFLINE_PATH.read_text(encoding="utf-8")
    assert "weight_syncer/patch_syncer@weight_syncer" in offline_text
    assert "actor,rollout: 0-0" in offline_text
    assert list(env.wrist_image_keys) == [
        "cam_left_wrist",
        "cam_right_wrist",
    ]
    assert env.main_image_key == "cam_high"
    assert env.override_cfg.is_dummy is True
    assert env.override_cfg.send_actions is False


def test_aloha_artifacts_use_environment_paths_and_keep_upstream_examples():
    portable_paths = [
        STAGE1_PATH,
        ONLINE_PATH,
        OFFLINE_PATH,
        ENV_PATH,
        SERVING_PATH,
        LEGACY_RUNTIME_PATH,
        LEGACY_REQUIREMENTS_PATH,
        REPO_ROOT / "examples/serving/scripts/run_serve_pi05_aloha.sh",
        REPO_ROOT / "examples/serving/scripts/serve_pi05_aloha.py",
        REPO_ROOT / "toolkits/replay_buffer/convert_hdf5_to_rlinf_buffer.py",
    ]
    absolute_machine_path = re.compile(r"/(?:home|root|mnt|workspace|inspire)(?:/|\b)")
    for path in portable_paths:
        assert not absolute_machine_path.search(path.read_text(encoding="utf-8")), path

    stage1_text = STAGE1_PATH.read_text(encoding="utf-8")
    assert "ALOHA_DATASET_PATH" in stage1_text
    assert "ALOHA_NORM_STATS_PATH" in stage1_text
    assert "ALOHA_STAGE1_CHECKPOINT" in ONLINE_PATH.read_text(encoding="utf-8")
    assert "ALOHA_REPLAY_BUFFER_PATH" in OFFLINE_PATH.read_text(encoding="utf-8")
    assert "ALOHA_STAGE1_CHECKPOINT" in SERVING_PATH.read_text(encoding="utf-8")

    assert (
        REPO_ROOT / "examples/sft/config/realworld_rlt_stage1_sft_openpi_pi05.yaml"
    ).is_file()
    assert (
        REPO_ROOT / "examples/embodiment/config/realworld_rlt_stage2_ac_mlp.yaml"
    ).is_file()


def test_legacy_runtime_snapshot_is_pinned_and_isolated():
    legacy = OmegaConf.load(LEGACY_RUNTIME_PATH)
    requirements_path = REPO_ROOT / legacy.dependency_snapshot.requirements
    assert requirements_path == LEGACY_REQUIREMENTS_PATH

    assert legacy.source.branch == "legacy/main-3eeb9265"
    assert legacy.source.commit == "3eeb9265e4574ef0c046ecf0cfee892e8cc6e9aa"
    assert legacy.checkpoint_contract.format == "legacy"
    assert legacy.checkpoint_contract.model_type == "openpi"
    assert legacy.checkpoint_contract.repo_id == "pi05_sandwich_merged_all_0805"
    assert legacy.checkpoint_contract.config_name == "pi05_aloha_robotwin"
    assert legacy.checkpoint_contract.action_horizon == 16
    assert legacy.checkpoint_contract.action_dim == 14
    assert legacy.checkpoint_contract.num_images_in_input == 3
    assert legacy.isolation.new_runtime_accepts_legacy is False
    assert legacy.isolation.reuse_integration_environment is False
    assert legacy.isolation.overwrite_legacy_checkpoint is False

    requirements = requirements_path.read_text(encoding="utf-8")
    assert "torch==2.11.0+cu128" in requirements
    assert "rlinf-openpi==0.1.1" in requirements
    assert "jax==0.5.3" in requirements
    assert "orbax-checkpoint==0.11.13" in requirements
