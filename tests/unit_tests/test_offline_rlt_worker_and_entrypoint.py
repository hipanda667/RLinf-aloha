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

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_WORKER_MODULE = "rlinf.workers.actor.fsdp_rlt_ac_policy_worker"
OFFLINE_WORKER_MODULE = "rlinf.workers.actor.fsdp_offline_rlt_ac_policy_worker"


def _load_isolated_offline_worker(monkeypatch):
    base_module = ModuleType(BASE_WORKER_MODULE)

    class FakeRLTACFSDPPolicy:
        def setup_sac_components(self):
            self.base_setup_called = True

    base_module.RLTACFSDPPolicy = FakeRLTACFSDPPolicy
    monkeypatch.setitem(sys.modules, BASE_WORKER_MODULE, base_module)

    module_name = "_test_fsdp_offline_rlt_ac_policy_worker"
    spec = importlib.util.spec_from_file_location(
        module_name,
        REPO_ROOT / "rlinf/workers/actor/fsdp_offline_rlt_ac_policy_worker.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeReplayBuffer:
    def __init__(self, stats):
        self.stats = stats
        self.load_calls = []

    def load_checkpoint(self, path, **kwargs):
        self.load_calls.append((path, kwargs))

    def get_stats(self):
        return self.stats


def test_offline_worker_loads_the_current_rank_shard(monkeypatch, tmp_path):
    module = _load_isolated_offline_worker(monkeypatch)
    replay = _FakeReplayBuffer(
        {
            "num_trajectories": 2,
            "total_samples": 32,
            "cache_size": 0,
        }
    )
    validation_calls = []
    monkeypatch.setattr(
        module,
        "validate_replay_checkpoint",
        lambda path, **kwargs: (
            validation_calls.append((path, kwargs))
            or SimpleNamespace(path=Path(path).resolve())
        ),
    )

    worker = module.OfflineRLTACFSDPPolicy.__new__(module.OfflineRLTACFSDPPolicy)
    worker.cfg = OmegaConf.create(
        {
            "algorithm": {
                "replay_buffer": {
                    "load_path": str(tmp_path),
                    "min_sample_count": 64,
                    "min_buffer_size": 1,
                }
            },
            "actor": {"global_batch_size": 32},
        }
    )
    worker.use_rlt_schedule = False
    worker._world_size = 2
    worker._rank = 1
    worker.replay_buffer = replay
    worker.log_info = lambda _message: None

    worker.setup_sac_components()

    assert worker.base_setup_called
    assert validation_calls == [
        (
            str(tmp_path),
            {
                "min_sample_count": 64,
                "world_size": 2,
            },
        )
    ]
    assert replay.load_calls == [
        (
            str(tmp_path.resolve()),
            {
                "is_distributed": True,
                "local_rank": 1,
                "world_size": 2,
            },
        )
    ]


def test_offline_worker_rejects_online_ingest_schedule(monkeypatch):
    module = _load_isolated_offline_worker(monkeypatch)
    worker = module.OfflineRLTACFSDPPolicy.__new__(module.OfflineRLTACFSDPPolicy)
    worker.use_rlt_schedule = True

    with pytest.raises(ValueError, match="does not ingest rollout transitions"):
        worker.setup_sac_components()


def test_pure_offline_entrypoint_launches_only_actor_workers(monkeypatch, tmp_path):
    from examples.embodiment import train_offline_rl
    from rlinf.data.storage import replay as replay_module

    events = []

    class FakeCluster:
        def __init__(self, cluster_cfg):
            events.append(("cluster", cluster_cfg.num_nodes))

    class FakePlacement:
        def __init__(self, _cfg, _cluster):
            pass

        def get_strategy(self, component):
            events.append(("strategy", component))
            assert component == "actor"
            return "actor-placement"

        def get_world_size(self, component):
            assert component == "actor"
            return 1

    class FakeActorGroup:
        worker_group_name = "ActorGroup"

    class FakeGroupFactory:
        def launch(self, _cluster, *, name, placement_strategy):
            events.append(("launch", name, placement_strategy))
            return FakeActorGroup()

    class FakeActorWorker:
        @classmethod
        def create_group(cls, _cfg):
            events.append(("create_group", cls.__name__))
            return FakeGroupFactory()

    worker_module = ModuleType(OFFLINE_WORKER_MODULE)
    worker_module.OfflineRLTACFSDPPolicy = FakeActorWorker
    monkeypatch.setitem(sys.modules, OFFLINE_WORKER_MODULE, worker_module)

    class FakeRunner:
        def __init__(self, *, cfg, actor, env, rollout):
            del cfg
            events.append(("runner", actor, env, rollout))

        def init_workers(self):
            events.append(("init_workers",))

        def run(self):
            events.append(("run",))

    validation_calls = []
    monkeypatch.setattr(train_offline_rl, "validate_cfg", lambda cfg: cfg)
    monkeypatch.setattr(train_offline_rl, "Cluster", FakeCluster)
    monkeypatch.setattr(
        train_offline_rl,
        "HybridComponentPlacement",
        FakePlacement,
    )
    monkeypatch.setattr(train_offline_rl, "OfflineRunner", FakeRunner)
    monkeypatch.setattr(
        replay_module,
        "validate_replay_checkpoint",
        lambda path, **kwargs: validation_calls.append((path, kwargs)),
    )

    cfg = OmegaConf.create(
        {
            "cluster": {
                "num_nodes": 1,
                "component_placement": {"actor": "0-0"},
            },
            "runner": {
                "only_eval": False,
                "val_check_interval": -1,
                "ckpt_path": None,
            },
            "algorithm": {
                "loss_type": "rlt_ac",
                "replay_buffer": {
                    "load_path": str(tmp_path),
                    "min_sample_count": 32,
                },
            },
            "actor": {
                "group_name": "ActorGroup",
                "global_batch_size": 32,
                "model": {"model_type": "rlt_mlp_policy"},
            },
        }
    )

    train_offline_rl.main.__wrapped__(cfg)

    assert validation_calls == [
        (
            str(tmp_path),
            {
                "min_sample_count": 32,
                "world_size": 1,
            },
        )
    ]
    assert [event[1] for event in events if event[0] == "strategy"] == ["actor"]
    runner_event = next(event for event in events if event[0] == "runner")
    assert runner_event[2:] == (None, None)
    assert events[-2:] == [("init_workers",), ("run",)]
