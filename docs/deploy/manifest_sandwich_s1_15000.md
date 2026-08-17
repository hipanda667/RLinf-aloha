# Deployment Manifest — RLinf Stage-1 OpenPI Pi0.5 RLT SFT (ALOHA "make a sandwich")

Created: 2026-08-14
Repo: `RLin_aloha` (formerly `RLinf-worktree-rltoken-anhao`)

## 1. Model checkpoint identity

Deployment checkpoint (Stage 1, step 15,000):

```
/inspire/hdd/global_user/czxs253130583/fangchuan/work/RL/output/RLtoken/stage1/
rlt_stage1_sft_pi05_sandwich_merged_all_0805_corrected_norm_16chunk_50k_bz16_h100_2gpu/
rlt_stage1_sft_pi05_sandwich_merged_all_0805_corrected_norm_16chunk_50k_bz16_h100_2gpu/
checkpoints/global_step_15000/actor/
```

| Artifact | SHA256 |
|---|---|
| `model_state_dict/full_weights.pt` | `56eff4953ebed65bc87de0d30637b1e763dd4e19d08fb81c54f18c0d74e4a2e0` |
| `pi05_sandwich_merged_all_0805/norm_stats.json` | `96f97295b4047b3ead4b7bbae22af68af7c9b1c71fedd1edf87eed663bb011fb` |
| `dcp_checkpoint/.metadata` (train-only) | `118c24c6f91d313fba43c8b5a944b45a5f57056ea62bb64f1f69c613f9116e64` |

Training curve at step 15000 (from `tensorboard/events`):
`vla_loss=0.00753`, `rlt_loss=0.3832`, `grad_norm=1.98`, `total=0.3908`.

## 2. Training config contract (must be reproduced at deployment)

Source: `examples/sft/config/rlt_stage1_sft_openpi_pi05_sandwich_merged_all_0805_16chunk_50k.yaml`

| Field | Value |
|---|---|
| `openpi_data.repo_id` | `pi05_sandwich_merged_all_0805` |
| `openpi_data.default_prompt` | `make a sandwich` |
| `model.model_type` | `openpi` |
| `model.action_dim` | 14 |
| `model.num_action_chunks` | 16 |
| `model.num_steps` | 4 |
| `model.openpi.config_name` | `pi05_aloha_robotwin` |
| `model.openpi.num_images_in_input` | 3 |
| `model.openpi.action_horizon` | 16 |
| `model.openpi.action_chunk` | 16 |
| `model.openpi.action_env_dim` | 14 |
| `model.openpi.noise_method` | `flow_noise` |
| `model.openpi.noise_params` | `[0.16, 0.12, 200]` |
| `model.openpi.joint_logprob` | `True` |
| `model.openpi.detach_critic_input` | `True` |
| `model.openpi.use_rlt` | `True` |
| `model.openpi.rlt_alpha` | 1.0 |
| `model.openpi.rlt_prefix_seq_len` | 1024 |
| `model.openpi.rlt_image_only` | `False` |
| `model.openpi.rlt_use_mask` | `True` |

Transform chain (identical at training and serving):

```
AlohaInputs(adapt_to_pi=True) -> DeltaActions(mask=[T×6,F,T×6,F]) -> Normalize(norm_stats)
  -> Pi0.5/RLT backbone (flow-noise)
  -> Unnormalize -> AbsoluteActions -> AlohaOutputs(adapt_to_pi=True)
```

## 3. Environment / code versions

| Component | Version / commit |
|---|---|
| RLinf repo | `RLin_aloha` @ `b9c275c98704951463cb946797b55702fae4fe13` (plus L1/L2 working-tree changes; to be committed) |
| OpenPI fork | `YushunXiang/OCL-openpi` @ `aca4d11eb6d571e9f9a73b20ebf6395ab85c1c52` |
| Python | 3.11.15 |
| torch | 2.6.0+cu124 |
| CUDA | 12.4 |
| numpy | 1.26.4 |
| safetensors | 0.8.0 |

## 4. Serving protocol contract

- Protocol: openpi-client WebSocket (`openpi_client.websocket_client_policy`).
- Default port: **8001** (`examples/serving/config/serve_pi05_aloha_sandwich.yaml`).
- Request observation (ALOHA keys, HWC images as sent by the real robot):
  - `images.cam_high` `[H,W,3]` uint8
  - `images.cam_left_wrist` / `images.cam_right_wrist` `[H,W,3]` uint8
  - `state` `[14]` float32 (unnormalized; normalization handled server-side)
  - `prompt` str (optional; fallback `make a sandwich`)
- Alternatively OpenPI keys: `observation/image` `[H,W,3]`, `observation/wrist_image` `[2,H,W,3]`, `observation/state` `[14]`.
- Response:
  - `actions` `[16,14]` float32 — **final ALOHA absolute actions** (already through
    AbsoluteActions + AlohaOutputs; do NOT apply delta/absolute conversion again).
- Client must use `action_horizon=16` (e.g. `ActionChunkBroker(action_horizon=16)`).
  The default OpenPI ALOHA client horizon is 25 and must not be used.

## 5. Chunk-boundary note

Training optimizes per-chunk trajectories; nothing explicitly forces
`chunk_k[-1] == chunk_{k+1}[0]`. Large boundary gaps in real-robot rollout are
typically caused by: (a) stale robot state at replan time, (b) double
delta/absolute conversion on the robot side, (c) wrong norm stats / repo_id, or
(d) client horizon != 16. Verify via action trace before blaming the checkpoint.

## 6. Git hygiene

- All code, configs, tools and tests are committed to `RLin_aloha`.
- Data / model artifacts (checkpoints, weights, norm_stats.json, datasets) are
  excluded from git and distributed via shared storage / artifact server with the
  SHA256 values above.
