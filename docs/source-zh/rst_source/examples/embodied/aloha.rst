ALOHA 三明治 RLT
================

.. figure:: https://tonyzhaozh.github.io/aloha/resources/algo.png
   :align: center
   :width: 90%

   ALOHA 双臂策略学习系统。图片来源：`ALOHA 项目 <https://tonyzhaozh.github.io/aloha/>`_。

使用本配方为双臂三明治制作任务微调 π₀.₅ 策略。你将运行 Stage-1 RLT SFT，把 ALOHA HDF5 episode 转换为 RLinf replay 格式，离线训练轻量级 Stage-2 actor-critic，并通过严格兼容性检查部署新格式 checkpoint。

概览
----

在训练、replay 转换与部署阶段保持同一套 16-step 动作和归一化契约。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 模型
      :text-align: center

      OpenPI_RLinf π₀.₅ · RLT MLP

   .. grid-item-card:: 算法
      :text-align: center

      RLT SFT · offline actor-critic

   .. grid-item-card:: 任务
      :text-align: center

      制作三明治

   .. grid-item-card:: 硬件
      :text-align: center

      双臂 ALOHA · 3 个 RGB 相机 · GPU

| **你将完成：** 准备数据 → 训练 Stage 1 → 转换 replay → 训练 Stage 2 → 严格加载并部署。
| **前置条件：** :doc:`安装 </rst_source/start/installation>` · ALOHA 数据集 · OpenPI 基础 checkpoint · 用于模型训练的 CUDA。

任务
~~~~

.. list-table::
   :header-rows: 1
   :widths: 18 40 42

   * - 阶段
     - 配置 / 入口
     - 结果
   * - Stage-1 SFT
     - ``aloha_sandwich_rlt_stage1_sft_openpi_pi05``
     - 使用 ALOHA 示教训练 π₀.₅ 及其 RLT feature decoder。
   * - Replay 转换
     - ``toolkits/replay_buffer/convert_hdf5_to_rlinf_buffer.py``
     - 生成经过校验的 ``metadata.json``、``trajectory_index.json`` 和 trajectory shard。
   * - Stage-2 offline RL
     - ``aloha_sandwich_rlt_stage2_offline_ac_mlp``
     - 仅训练 RLT MLP actor-critic，不创建 environment 或 rollout worker。
   * - 策略部署
     - ``examples/serving/scripts/serve_pi05_aloha.py``
     - 严格加载新的 ``openpi_rlinf`` checkpoint，并提供 WebSocket policy protocol。

观测与动作
~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - 字段
     - 契约
   * - Observation
     - 一张 ``cam_high`` RGB 图像，按顺序排列的 ``cam_left_wrist`` 与 ``cam_right_wrist`` RGB 图像，以及 14D follower joint state。
   * - Action
     - 16-step 的绝对 14D 双臂关节与夹爪 target chunk。
   * - Reward
     - 显式的三明治 success / failure / abort / timeout 标签；未标注 step 的 reward 为零。
   * - Prompt
     - ``make a sandwich``，归一化资产由 ``repo_id: aloha_sandwich`` 选择。

安装
----

.. include:: _setup_common.rst

安装 OpenPI stack 与轻量级 dummy environment 依赖：

.. code-block:: bash

   bash requirements/install.sh embodied --model openpi --env maniskill_libero
   source .venv/bin/activate

此命令会创建 RLinf 环境，安装 ``openpi_rlinf`` 使用的 OpenPI runtime，并安装 upstream RLT 测试使用的 OpenPI/ManiSkill-LIBERO 依赖组合。ALOHA adapter 默认仍为 no-motion。机器人端进程所需的 ROS 与 Interbotix 依赖请按照 `ALOHA 官方说明 <https://github.com/tonyzhaozh/aloha>`_ 安装。

准备模型和数据
--------------

让可移植配置指向你的本地资产。在所有阶段保持 ``repo_id``、action dimension 与 16-step horizon 不变。

.. code-block:: bash

   export OPENPI_MODEL_PATH=/path/to/openpi-base-checkpoint
   export ALOHA_DATASET_PATH=/path/to/lerobot-aloha-sandwich
   export ALOHA_NORM_STATS_PATH=/path/to/lerobot-aloha-sandwich/norm_stats.json
   export ALOHA_HDF5_DIR=/path/to/aloha-hdf5-episodes
   export RLINF_OUTPUT_DIR=/path/to/rlinf-results

LeRobot 数据集必须包含三个具名 camera stream、14D state/action 数据，以及 ``aloha_sandwich`` 的 normalization statistics。

运行
----

训练 Stage 1：

.. code-block:: bash

   bash examples/sft/run_vla_sft.sh \
      aloha_sandwich_rlt_stage1_sft_openpi_pi05

此命令使用 ``action_horizon=16`` 和联合 RLT reconstruction objective 训练新的 PyTorch ``openpi_rlinf`` π₀.₅ wrapper。把 ``ALOHA_STAGE1_CHECKPOINT`` 设置为生成的 actor checkpoint 目录。

转换 offline episode：

.. code-block:: bash

   export ALOHA_STAGE1_CHECKPOINT=/path/to/stage1/checkpoint/actor
   export ALOHA_REPLAY_BUFFER_PATH=/path/to/new/replay-buffer
   python toolkits/replay_buffer/convert_hdf5_to_rlinf_buffer.py

此命令使用冻结的 Stage-1 模型提取 ``z_rl``、proprioception 和 reference chunk，写入新的 replay checkpoint，并校验其 metadata、index、sample count 与 trajectory 文件。转换器拒绝覆盖已存在的输出目录。

在没有 robot 或 rollout worker 的情况下训练 Stage 2：

.. code-block:: bash

   bash examples/embodiment/run_offline_rl.sh \
      aloha_sandwich_rlt_stage2_offline_ac_mlp

此命令把完整 trajectory 分配给各 actor rank，并从 replay buffer 训练 RLT MLP actor-critic。使用 ``runner.resume_dir=/path/to/global_step_N`` 恢复完整的 Stage-2 run；不要把 Stage-1 checkpoint 写入 ``runner.ckpt_path``。

在单独的 deployment port 上部署新格式 Stage-1 policy：

.. code-block:: bash

   bash examples/serving/scripts/run_serve_pi05_aloha.sh \
      --config examples/serving/config/serve_pi05_aloha_sandwich.yaml \
      --smoke-test

Smoke test 要求三个 camera input，拒绝非有限值或非 ``(16, 14)`` 的 action，并校验 Stage-2 feature shape：``z_rl=(1, 2048)``、``proprio=(1, 14)`` 和 ``ref_chunk=(1, 16, 14)``。在 metadata 确认目标 checkpoint、``repo_id``、norm stats、horizon 和 Git commit 后，才可移除 ``--smoke-test``。

隔离旧 Checkpoint
~~~~~~~~~~~~~~~~~~~~~~~

仅在独立 worktree 与 virtual environment 中重建经过验证的旧服务：

.. code-block:: bash

   export RLINF_INTEGRATION_REPO=/path/to/RLinf-integration
   git worktree add --detach ../RLinf-aloha-legacy \
      3eeb9265e4574ef0c046ecf0cfee892e8cc6e9aa
   cd ../RLinf-aloha-legacy
   bash requirements/install.sh embodied --model openpi \
      --env maniskill_libero --venv .venv-aloha-legacy
   source .venv-aloha-legacy/bin/activate
   python -m pip install -r \
      "$RLINF_INTEGRATION_REPO/requirements/legacy/aloha_pi05_legacy_runtime.txt"
   export SERVING_PYTHON_BIN="$PWD/.venv-aloha-legacy/bin/python"
   bash examples/serving/scripts/run_serve_pi05_aloha.sh \
      --checkpoint-dir /path/to/legacy/checkpoint/actor \
      --repo-id pi05_sandwich_merged_all_0805 \
      --strict-load --smoke-test

此流程会在不移动任何 branch 的前提下 checkout 精确的旧 source，把经过验证的 dependency snapshot 安装到专用环境，覆盖历史 machine-specific checkpoint path，并要求执行 strict smoke test。不可变的 source/configuration identity 记录在 ``examples/serving/config/legacy_pi05_aloha_runtime.yaml`` 中。切勿让 ``SERVING_PYTHON_BIN`` 指向 integration environment。

.. warning::

   旧 Stage-1 checkpoint 使用不同的 RLT decoder layout。请继续使用保留的 ``legacy/main-3eeb9265`` branch 与原始环境部署旧 checkpoint。本 runtime 会拒绝 ``checkpoint.format: legacy``，且绝不会回退到 non-strict load。

.. warning::

   当前 upstream integration 仅提供 deterministic no-motion ALOHA backend。在 hardware backend 实现并验证 joint limit、单步 delta limit、watchdog 与 emergency stop 前，real action sending 始终锁定。暂时不要覆盖 ``realworld_aloha_sandwich.yaml`` 中的 ``is_dummy: True`` 或 ``send_actions: False``。

可视化与结果
------------

在 TensorBoard 中观察 Stage-1 loss 与 Stage-2 actor/critic metric。指标定义见 :doc:`训练指标 <../../reference/metrics>`。训练 Stage 2 前，请检查 ``conversion_manifest.json``；其中记录了 source episode，以及 Stage-1 weight 与 norm stats 的 SHA256 identity。

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - 检查项
     - 预期结果
   * - Stage-1 output
     - Action shape 为 ``[batch, 16, 14]``，且全部为有限值。
   * - Replay checkpoint
     - Metadata 与 index count 一致，每个 trajectory 文件存在，且每个 actor rank 都获得非空 shard。
   * - Policy smoke test
     - Metadata 报告 ``model_type: openpi_rlinf``、``checkpoint_format: openpi_rlinf``、``repo_id: aloha_sandwich`` 与 ``action_horizon: 16``；Stage-2 feature shape 为 ``(1, 2048)``、``(1, 14)`` 和 ``(1, 16, 14)``。

存储语义见 :doc:`Replay Buffer <../../concepts/replay_buffer>`，完整 run 的 checkpoint 恢复方法见 :doc:`恢复训练 <../../guides/resume>`。
