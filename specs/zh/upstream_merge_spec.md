# Upstream 合并规范

## 状态

- 状态：提议
- 评估日期：2026-08-17
- 本地基线：`main@3eeb9265`
- Upstream 基线：`upstream/main@9ad44393`
- 共同祖先：`d0a97caa`
- 分叉情况：本地独有 4 个提交，upstream 独有 62 个提交

## 决策

需要同步 `upstream/main`，但禁止直接在当前 `main` 上执行无条件 merge。

集成必须从 `upstream/main@9ad44393` 创建独立分支，再选择性迁移本地 ALOHA、π0.5 SFT、RLT Stage 2 离线训练和部署能力。现有 Stage-1 checkpoint 必须继续由旧运行时提供服务，直到完成兼容层验证或使用新的 `openpi_rlinf` 实现重新训练。

## 背景

本地 `main` 包含以下独有提交：

| 提交 | 内容 | 集成策略 |
| --- | --- | --- |
| `edd699cf` | RLT Stage-1 SFT 和 Stage-2 离线训练 | 不整体 cherry-pick；只迁移 upstream 尚未提供的离线训练能力 |
| `b9c275c9` | ALOHA 真机、sandwich 任务和合约测试 | 迁移并适配 upstream 的 RealWorld 接口 |
| `7c0632e8` | π0.5 RLT Stage-1 policy server 和 ALOHA 配置 | 迁移；旧 checkpoint 路径保留旧模型实现 |
| `3eeb9265` | README 中的 RLT 复现指南 | 根据新路径和新模型类型重写，不直接保留旧命令 |

Upstream 已提供新的 RLT 主线实现和相关修复：

| 提交 | 影响 |
| --- | --- |
| `3d93750d` | 增加 ManiSkill RLT Stage 1/2、真实机器人配置、测试和文档 |
| `c704688c` | 将 RLT 迁移到与 JAX 对齐的 PyTorch π0.5，即 `openpi_rlinf` |
| `8587bac4` | 修复 RLT reconstruction 和 intervention reference-action 逻辑 |
| `13e5b652` | 增加 RLT TD3 MLP Stage-2 策略 |
| `5fb8b074` | 增加 LeRobot 数据格式兼容层 |
| `d3aff547` | 重构 `rlinf.data` 的 schema、storage 和 dataset 模块 |
| `9ad44393` | 将 `EmbodiedFSDPActor` 拆分到独立模块 |

## 目标

1. 采用 upstream 的 π0.5 SFT、RLT reconstruction、RLT actor 和数据模块实现。
2. 保留 ALOHA sandwich 的数据、训练、在线控制和部署能力。
3. 保留纯离线 RLT Stage-2 训练能力，并迁移到 upstream 当前接口。
4. 防止旧 Stage-1 checkpoint 被新 decoder 结构静默错误加载。
5. 在合入 `main` 前完成 CPU 单元测试、配置检查和 GPU smoke/E2E 测试。

## 非目标

- 本次集成不要求把 ALOHA 作为新的 `SupportedEnvType` 注册；继续通过 `realworld` 环境集成，除非后续设计另有决定。
- 本次集成不承诺旧 Stage-1 checkpoint 可以被 `openpi_rlinf` 直接加载。
- 本次集成不包含与 ALOHA、π0.5 SFT 或 RLT 无关的 upstream 功能验收。
- 不在迁移过程中修改或覆盖现有 checkpoint、数据集和实验结果。

## 兼容性约束

### Stage-1 Checkpoint

`8587bac4` 修改了 RLT token decoder 的参数结构和 reconstruction 目标。旧 checkpoint 包含旧 decoder 的参数命名和形状，不能假设与新实现兼容。

必须采用以下策略之一：

1. 旧运行时隔离，推荐作为迁移期间的默认方案。
   - 使用当前 `main@3eeb9265` 或固定提交运行旧 policy server。
   - 保存对应 Python 环境、OpenPI 依赖和配置快照。
   - 不允许新代码覆盖旧部署环境。
2. 使用 `openpi_rlinf` 重新训练 Stage 1，作为目标方案。
   - 使用 upstream 的 autoregressive reconstruction。
   - 重新生成 checkpoint，并完成 Stage-1 loss、推理 action shape 和 Stage-2 feature extraction 验证。
3. 实现显式 legacy loader，仅在无法重新训练时采用。
   - legacy 模型必须使用独立类名或明确的配置开关。
   - 禁止通过 `strict=False` 忽略缺失或多余参数。
   - 必须添加旧 checkpoint 加载和固定输入推理测试。

### ALOHA Action Horizon

ALOHA sandwich 使用 `action_horizon=16`。本地 `FSDPVlaSftWorker` 的补丁会把 `actor.model.openpi.action_horizon` 传入官方 OpenPI dataloader，而 upstream 的 legacy `openpi` loader 当前没有保留该覆盖逻辑。

集成时必须：

- 以 upstream 的 `rlinf/data/datasets/openpi_rlinf/official_sft_data_loader.py` 为基础；
- 对 legacy `model_type: openpi` 保留显式 `action_horizon` 覆盖；
- 不放宽 `model_type: openpi_rlinf` 的 shape validation；
- 添加测试，确认 dataset action horizon、model horizon 和输出 chunk 均为 16。

### 安装环境

Upstream Docker 默认 CUDA 从 12.4 升级到 12.8。集成环境必须重新构建，不能复用未经验证的旧 venv 或 Docker layer。

`openpi_rlinf` 仍通过以下安装目标安装：

```bash
bash requirements/install.sh embodied --model openpi --env maniskill_libero
```

## 集成方式

### 1. 创建集成分支

不要在当前 `main` 上直接 merge。创建以已审核 upstream 提交为基线的分支：

```bash
git fetch upstream main
git switch -c integrate/upstream-20260815 upstream/main
```

集成开始前，为旧部署创建不可变 tag 或保留分支，并记录 checkpoint 使用的提交、配置和依赖环境。

### 2. 迁移本地能力

按以下顺序迁移，且每一步单独提交：

1. ALOHA environment adapter、hardware contract、task reward 和单元测试。
2. ALOHA sandwich 配置，基于 upstream 已重命名的 real-world RLT 配置重新创建。
3. π0.5 ALOHA policy server，并明确选择 legacy 或 `openpi_rlinf` checkpoint 格式。
4. 离线 replay-buffer 转换工具，更新到新的 data schema/storage 导入路径。
5. 离线 RLT Stage-2 worker 和 runner，继承 upstream 当前 RLT actor。
6. README/spec 文档和操作命令。

不得整体 cherry-pick `edd699cf`，因为其中部分 RLT 基础实现已被 upstream 替代。

### 3. 必须迁移的接口

| 旧接口 | 新接口 | 要求 |
| --- | --- | --- |
| `rlinf.workers.actor.rlt_ac_policy_worker` | `rlinf.workers.actor.fsdp_rlt_ac_policy_worker` | `OfflineRLTACFSDPPolicy` 继承新的 `RLTACFSDPPolicy` |
| `rlinf.data.embodied_io_struct` | `rlinf.data.schema` | 使用新 schema 导出的 trajectory 类型 |
| `rlinf.data.replay_buffer` | `rlinf.data.storage.replay` | 使用新的 replay-buffer 导出路径 |
| `rlinf.data.lerobot_paths` | `rlinf.data.storage.lerobot` | 更新 smoke 和数据工具导入 |
| `examples/sft/config/rlt_stage1_sft_openpi_pi05.yaml` | `examples/sft/config/realworld_rlt_stage1_sft_openpi_pi05.yaml` | ALOHA 配置使用独立名称，不能覆盖 Franka 示例 |
| `examples/embodiment/config/rlt_stage2_ac_mlp.yaml` | `examples/embodiment/config/realworld_rlt_stage2_ac_mlp.yaml` | ALOHA 配置使用独立名称，不能覆盖通用 real-world 示例 |

### 4. 冲突解决规则

虚拟 merge 已确认以下冲突：

| 文件 | 冲突类型 | 解决规则 |
| --- | --- | --- |
| `examples/sft/config/realworld_rlt_stage1_sft_openpi_pi05.yaml` | rename/content | 保留 upstream Franka 示例；新增单独的 ALOHA sandwich 配置 |
| `examples/embodiment/config/realworld_rlt_stage2_ac_mlp.yaml` | rename/content | 保留 upstream 示例；新增 ALOHA online/offline 配置 |
| `rlinf/workers/sft/fsdp_vla_sft_worker.py` | content | 采用 upstream worker；将 horizon 覆盖移入新的 dataloader helper 并测试 |
| `toolkits/lerobot/calculate_norm_stats.py` | modify/delete | 保留 upstream 文件和兼容修复；单独迁移本地新增能力，不恢复旧文件版本 |

以下自动合并文件仍必须人工复审：

- `rlinf/envs/realworld/realworld_env.py`
- `rlinf/models/embodiment/mlp_policy/__init__.py`
- `.gitignore`
- `README.md`

自动合并成功不代表接口或运行时语义兼容。

## 实现要求

### 离线 RLT Stage 2

- `OfflineRLTACFSDPPolicy` 必须继承 upstream 的 `RLTACFSDPPolicy`。
- replay buffer 加载必须使用 upstream 当前的 `load_checkpoint` API。
- 每个 distributed rank 都必须加载正确的 shard。
- 启动前必须校验 `metadata.json`、`trajectory_index.json` 和最小样本数。
- 纯离线模式不得创建 env 或 rollout worker。
- checkpoint resume 必须使用完整的 `runner.resume_dir`，不能把 Stage-1 checkpoint 误作 Stage-2 actor checkpoint。

### Policy Server

- 服务端必须在 metadata 中报告模型类型、checkpoint、norm stats、action horizon 和 Git commit。
- state dict 默认 strict load；禁止静默忽略参数不匹配。
- norm stats 必须来自 checkpoint 对应的 `repo_id`，禁止回退到其他数据集。
- legacy 和 `openpi_rlinf` 加载路径必须显式区分。
- smoke test 必须验证 action shape、finite values 和三路 ALOHA 图像输入。

### 配置

- 禁止提交机器专用的绝对路径。
- ALOHA 配置中的数据、模型、checkpoint 和输出目录必须使用占位路径或环境变量。
- `repo_id`、`config_name`、`action_dim`、`action_horizon` 和 norm stats 必须在 Stage 1、部署和 Stage 2 之间一致。
- 新增配置不得覆盖 upstream 的 Franka 或 ManiSkill 示例。

## 验收标准

### 静态和 CPU 检查

以下检查必须通过：

```bash
python -m compileall rlinf examples toolkits
pre-commit run --all-files
pytest -q tests/unit_tests/test_aloha_realworld_contract.py
pytest -q tests/unit_tests/test_convert_hdf5_to_rlinf_buffer.py
pytest -q tests/unit_tests/test_rlt_mlp_policy.py
pytest -q tests/unit_tests/test_rlt_token_transformer.py
```

还必须验证：

- 仓库中不存在对已删除 data/RLT 模块的导入；
- Hydra 可以组合 ALOHA Stage-1、Stage-2 online 和 Stage-2 offline 配置；
- 中英文文档中的配置名、命令、模型类型和路径一致。

### GPU 和运行时检查

合入 `main` 前必须完成：

1. π0.5 ALOHA Stage-1 单步训练，确认 `vla_loss` 和 `rlt_loss` 均为 finite。
2. 新 checkpoint strict-load 和单次 policy-server 推理。
3. Stage-2 feature extraction，确认 `z_rl`、`proprio` 和 `ref_chunk` shape。
4. 离线 replay buffer 在单 GPU 上完成至少一次 actor/critic update。
5. upstream 的 ManiSkill RLT Stage-1 和 Stage-2 E2E 不发生回归。
6. 如保留旧 checkpoint 服务，旧运行时完成一次真实 checkpoint smoke test。

### 合并门槛

只有同时满足以下条件才能合入本地 `main`：

- 所有 Git 冲突已按本规范解决；
- 不存在已删除模块的运行时导入；
- 旧 checkpoint 已隔离，或兼容方案已通过 strict-load 测试；
- ALOHA 16-step action horizon 有自动化测试；
- CPU 检查和要求的 GPU smoke/E2E 均通过；
- 安装和运行步骤已在干净环境中复现。

## 回滚方案

- 在集成前保留当前 `main@3eeb9265` 的分支或 tag。
- 旧 Stage-1 policy server 继续固定到旧提交和旧环境，直到新 checkpoint 验收完成。
- 新代码通过独立部署端口和配置进行灰度验证，不覆盖旧服务。
- 若 Stage-1 loss、action shape、Stage-2 reward curve 或真机行为异常，停止合入并回到旧部署；不得通过关闭 strict loading 规避问题。

## 已知 Upstream 文档问题

Upstream 的中英文 RLT 文档在 Stage-1 编辑示例中都曾把 `openpi_data` 放在 `actor` 下。实际配置要求 `actor.model.openpi_data`。迁移文档时必须使用真实配置层级，并保持中英文一致。

## 完成定义

当集成分支通过全部验收门槛、旧 checkpoint 的运行方式已明确记录、新 ALOHA checkpoint 可以用于 Stage-1 部署和 Stage-2 特征提取时，本规范视为完成。此后可以通过一个经过审核的 PR 将集成分支合入本地 `main`。
