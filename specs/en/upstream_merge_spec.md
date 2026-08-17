# Upstream Merge Specification

## Status

- Status: Proposed
- Assessment date: 2026-08-17
- Local baseline: `main@3eeb9265`
- Upstream baseline: `upstream/main@9ad44393`
- Merge base: `d0a97caa`
- Divergence: 4 local-only commits and 62 upstream-only commits

## Decision

Synchronize with `upstream/main`, but do not perform an unconditional merge into the current `main` branch.

Create a dedicated integration branch from `upstream/main@9ad44393`, then selectively port the local ALOHA, π0.5 SFT, offline RLT Stage 2, and serving capabilities. Continue serving the existing Stage-1 checkpoint with the legacy runtime until a compatibility layer is verified or Stage 1 is retrained with the new `openpi_rlinf` implementation.

## Background

The local `main` branch contains these unique commits:

| Commit | Content | Integration strategy |
| --- | --- | --- |
| `edd699cf` | RLT Stage-1 SFT and offline Stage-2 training | Do not cherry-pick as a whole; port only offline training capabilities not supplied by upstream |
| `b9c275c9` | ALOHA hardware support, sandwich task, and contract tests | Port and adapt to the upstream RealWorld interfaces |
| `7c0632e8` | π0.5 RLT Stage-1 policy server and ALOHA configuration | Port; retain the legacy model path for old checkpoints |
| `3eeb9265` | RLT reproduction guide in README | Rewrite for the new paths and model types instead of retaining old commands verbatim |

Upstream now supplies the primary RLT implementation and relevant fixes:

| Commit | Impact |
| --- | --- |
| `3d93750d` | Adds ManiSkill RLT Stage 1/2, real-robot configs, tests, and docs |
| `c704688c` | Migrates RLT to the JAX-aligned PyTorch π0.5 implementation, `openpi_rlinf` |
| `8587bac4` | Fixes RLT reconstruction and intervention reference-action handling |
| `13e5b652` | Adds the RLT TD3 MLP Stage-2 policy |
| `5fb8b074` | Adds a LeRobot data-format compatibility layer |
| `d3aff547` | Restructures `rlinf.data` into schema, storage, and dataset modules |
| `9ad44393` | Splits `EmbodiedFSDPActor` into a dedicated module |

## Goals

1. Adopt the upstream π0.5 SFT, RLT reconstruction, RLT actor, and data-module implementations.
2. Preserve ALOHA sandwich data, training, online control, and serving capabilities.
3. Preserve pure offline RLT Stage-2 training and migrate it to current upstream interfaces.
4. Prevent old Stage-1 checkpoints from being silently loaded by the new decoder structure.
5. Complete CPU unit tests, configuration checks, and GPU smoke/E2E tests before merging into `main`.

## Non-Goals

- This integration does not require registering ALOHA as a new `SupportedEnvType`; continue integrating through `realworld` unless a later design decides otherwise.
- This integration does not guarantee that an old Stage-1 checkpoint can be loaded directly by `openpi_rlinf`.
- This integration does not validate upstream features unrelated to ALOHA, π0.5 SFT, or RLT.
- Do not modify or overwrite existing checkpoints, datasets, or experiment results during migration.

## Compatibility Constraints

### Stage-1 Checkpoints

Commit `8587bac4` changes the parameter structure and reconstruction objective of the RLT token decoder. Old checkpoints contain parameter names and shapes from the previous decoder and must not be assumed compatible with the new implementation.

Use one of these strategies:

1. Isolate the legacy runtime, recommended during migration.
   - Serve old checkpoints from the current `main@3eeb9265` or another pinned commit.
   - Preserve the corresponding Python environment, OpenPI dependencies, and configuration snapshot.
   - Do not allow the new code to overwrite the legacy deployment environment.
2. Retrain Stage 1 with `openpi_rlinf`, the target solution.
   - Use the upstream autoregressive reconstruction objective.
   - Produce a new checkpoint and validate Stage-1 losses, inference action shape, and Stage-2 feature extraction.
3. Implement an explicit legacy loader only when retraining is not possible.
   - Use a separate class name or an explicit configuration switch.
   - Do not suppress missing or unexpected parameters with `strict=False`.
   - Add an old-checkpoint loading test and a fixed-input inference test.

### ALOHA Action Horizon

ALOHA sandwich uses `action_horizon=16`. The local `FSDPVlaSftWorker` patch forwards `actor.model.openpi.action_horizon` to the official OpenPI dataloader. The upstream legacy `openpi` loader does not currently retain this override.

The integration must:

- use upstream `rlinf/data/datasets/openpi_rlinf/official_sft_data_loader.py` as the base;
- preserve an explicit `action_horizon` override for legacy `model_type: openpi`;
- retain shape validation for `model_type: openpi_rlinf`;
- add a test proving that the dataset horizon, model horizon, and output chunk are all 16.

### Installation Environment

The upstream Docker default moves from CUDA 12.4 to CUDA 12.8. Rebuild the integration environment instead of reusing an unverified venv or Docker layer.

Install `openpi_rlinf` through the existing OpenPI target:

```bash
bash requirements/install.sh embodied --model openpi --env maniskill_libero
```

## Integration Method

### 1. Create an Integration Branch

Do not merge directly into the current `main`. Create a branch from the reviewed upstream commit:

```bash
git fetch upstream main
git switch -c integrate/upstream-20260815 upstream/main
```

Before integration, create an immutable tag or retain a branch for the legacy deployment. Record the commit, configuration, and dependency environment used by each checkpoint.

### 2. Port Local Capabilities

Port the following items in order and commit each step separately:

1. ALOHA environment adapter, hardware contract, task reward, and unit tests.
2. ALOHA sandwich configs recreated from the renamed upstream real-world RLT configs.
3. The π0.5 ALOHA policy server with an explicit legacy or `openpi_rlinf` checkpoint format.
4. Offline replay-buffer conversion tools updated to the new data schema/storage paths.
5. Offline RLT Stage-2 worker and runner based on the current upstream RLT actor.
6. README/spec documentation and operating commands.

Do not cherry-pick `edd699cf` as a whole because upstream supersedes part of its foundational RLT implementation.

### 3. Required API Migrations

| Old interface | New interface | Requirement |
| --- | --- | --- |
| `rlinf.workers.actor.rlt_ac_policy_worker` | `rlinf.workers.actor.fsdp_rlt_ac_policy_worker` | Derive `OfflineRLTACFSDPPolicy` from the new `RLTACFSDPPolicy` |
| `rlinf.data.embodied_io_struct` | `rlinf.data.schema` | Use trajectory types exported by the new schema |
| `rlinf.data.replay_buffer` | `rlinf.data.storage.replay` | Use the new replay-buffer export path |
| `rlinf.data.lerobot_paths` | `rlinf.data.storage.lerobot` | Update smoke-test and data-tool imports |
| `examples/sft/config/rlt_stage1_sft_openpi_pi05.yaml` | `examples/sft/config/realworld_rlt_stage1_sft_openpi_pi05.yaml` | Give the ALOHA config a distinct name; do not overwrite the Franka example |
| `examples/embodiment/config/rlt_stage2_ac_mlp.yaml` | `examples/embodiment/config/realworld_rlt_stage2_ac_mlp.yaml` | Give ALOHA configs distinct names; do not overwrite the generic real-world example |

### 4. Conflict Resolution Rules

The virtual merge confirmed these conflicts:

| File | Conflict type | Resolution rule |
| --- | --- | --- |
| `examples/sft/config/realworld_rlt_stage1_sft_openpi_pi05.yaml` | rename/content | Keep the upstream Franka example and add a separate ALOHA sandwich config |
| `examples/embodiment/config/realworld_rlt_stage2_ac_mlp.yaml` | rename/content | Keep the upstream example and add separate ALOHA online/offline configs |
| `rlinf/workers/sft/fsdp_vla_sft_worker.py` | content | Keep the upstream worker; move the horizon override into the new dataloader helper and test it |
| `toolkits/lerobot/calculate_norm_stats.py` | modify/delete | Keep the upstream file and compatibility fixes; port local additions separately instead of restoring the old file |

Manually review these automatically merged files:

- `rlinf/envs/realworld/realworld_env.py`
- `rlinf/models/embodiment/mlp_policy/__init__.py`
- `.gitignore`
- `README.md`

A clean textual merge does not prove API or runtime semantic compatibility.

## Implementation Requirements

### Offline RLT Stage 2

- Derive `OfflineRLTACFSDPPolicy` from the upstream `RLTACFSDPPolicy`.
- Load replay buffers through the current upstream `load_checkpoint` API.
- Load the correct shard on every distributed rank.
- Validate `metadata.json`, `trajectory_index.json`, and the minimum sample count before training.
- Do not create environment or rollout workers in pure offline mode.
- Resume complete Stage-2 runs through `runner.resume_dir`; do not treat a Stage-1 checkpoint as a Stage-2 actor checkpoint.

### Policy Server

- Report model type, checkpoint, norm stats, action horizon, and Git commit in server metadata.
- Use strict state-dict loading by default; never silently ignore incompatible parameters.
- Load norm stats from the checkpoint's matching `repo_id`; do not fall back to another dataset.
- Explicitly distinguish legacy and `openpi_rlinf` loading paths.
- Verify action shape, finite values, and all three ALOHA image inputs in the smoke test.

### Configuration

- Do not commit machine-specific absolute paths.
- Use placeholder paths or environment variables for ALOHA data, models, checkpoints, and output directories.
- Keep `repo_id`, `config_name`, `action_dim`, `action_horizon`, and norm stats consistent across Stage 1, serving, and Stage 2.
- Do not overwrite upstream Franka or ManiSkill examples with new configs.

## Acceptance Criteria

### Static and CPU Checks

The following checks must pass:

```bash
python -m compileall rlinf examples toolkits
pre-commit run --all-files
pytest -q tests/unit_tests/test_aloha_realworld_contract.py
pytest -q tests/unit_tests/test_convert_hdf5_to_rlinf_buffer.py
pytest -q tests/unit_tests/test_rlt_mlp_policy.py
pytest -q tests/unit_tests/test_rlt_token_transformer.py
```

Also verify that:

- no code imports removed data or RLT modules;
- Hydra composes the ALOHA Stage-1, Stage-2 online, and Stage-2 offline configs;
- English and Chinese docs use matching config names, commands, model types, and paths.

### GPU and Runtime Checks

Complete these checks before merging into `main`:

1. Run one π0.5 ALOHA Stage-1 training step and confirm finite `vla_loss` and `rlt_loss`.
2. Strict-load a new checkpoint and run one policy-server inference.
3. Extract Stage-2 features and verify the shapes of `z_rl`, `proprio`, and `ref_chunk`.
4. Complete at least one actor/critic update from an offline replay buffer on one GPU.
5. Confirm no regression in the upstream ManiSkill RLT Stage-1 and Stage-2 E2E tests.
6. If legacy checkpoint serving remains supported, run one smoke test with a real legacy checkpoint in the legacy runtime.

### Merge Gate

Merge into the local `main` only when all of these conditions hold:

- all Git conflicts are resolved according to this specification;
- no runtime imports reference removed modules;
- old checkpoints are isolated or a compatibility solution passes strict-load tests;
- the ALOHA 16-step action horizon has automated coverage;
- all required CPU checks and GPU smoke/E2E tests pass;
- installation and runtime steps are reproduced in a clean environment.

## Rollback Plan

- Preserve a branch or tag for the current `main@3eeb9265` before integration.
- Keep the old Stage-1 policy server pinned to the old commit and environment until the new checkpoint passes acceptance.
- Validate the new code on a separate deployment port and config without replacing the old service.
- If Stage-1 losses, action shapes, Stage-2 reward curves, or real-robot behavior regress, stop the merge and return to the legacy deployment. Do not bypass the failure by disabling strict loading.

## Known Upstream Documentation Issue

The upstream English and Chinese RLT docs have shown `openpi_data` directly under `actor` in one Stage-1 editing example. The actual configuration requires `actor.model.openpi_data`. Use the real configuration hierarchy in migrated docs and keep both languages aligned.

## Definition of Done

This specification is complete when the integration branch passes every merge gate, the legacy checkpoint runtime is documented, and a new ALOHA checkpoint supports both Stage-1 serving and Stage-2 feature extraction. The integration branch may then be merged into the local `main` through a reviewed PR.
