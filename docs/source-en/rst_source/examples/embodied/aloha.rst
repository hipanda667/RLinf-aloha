ALOHA Sandwich RLT
==================

.. figure:: https://tonyzhaozh.github.io/aloha/resources/algo.png
   :align: center
   :width: 90%

   The ALOHA bimanual policy-learning setup. Image credit: the `ALOHA project <https://tonyzhaozh.github.io/aloha/>`_.

Use this recipe to fine-tune a π₀.₅ policy for dual-arm sandwich making. You will run Stage-1 RLT SFT, convert ALOHA HDF5 episodes into the RLinf replay format, train the lightweight Stage-2 actor-critic offline, and serve a new-format checkpoint with strict compatibility checks.

Overview
--------

Keep one 16-step action and normalization contract across training, replay conversion, and serving.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Models
      :text-align: center

      OpenPI_RLinf π₀.₅ · RLT MLP

   .. grid-item-card:: Algorithms
      :text-align: center

      RLT SFT · offline actor-critic

   .. grid-item-card:: Tasks
      :text-align: center

      Make a sandwich

   .. grid-item-card:: Hardware
      :text-align: center

      Dual-arm ALOHA · 3 RGB cameras · GPU

| **You'll do:** prepare data → train Stage 1 → convert replay → train Stage 2 → strict-load and serve.
| **Prerequisites:** :doc:`Installation </rst_source/start/installation>` · an ALOHA dataset · an OpenPI base checkpoint · CUDA for model training.

Tasks
~~~~~

.. list-table::
   :header-rows: 1
   :widths: 18 40 42

   * - Stage
     - Config / entry point
     - Outcome
   * - Stage-1 SFT
     - ``aloha_sandwich_rlt_stage1_sft_openpi_pi05``
     - Train π₀.₅ and its RLT feature decoder from ALOHA demonstrations.
   * - Replay conversion
     - ``toolkits/replay_buffer/convert_hdf5_to_rlinf_buffer.py``
     - Produce validated ``metadata.json``, ``trajectory_index.json``, and trajectory shards.
   * - Stage-2 offline RL
     - ``aloha_sandwich_rlt_stage2_offline_ac_mlp``
     - Train only the RLT MLP actor-critic; no environment or rollout workers are created.
   * - Policy serving
     - ``examples/serving/scripts/serve_pi05_aloha.py``
     - Strict-load a new ``openpi_rlinf`` checkpoint and expose the WebSocket policy protocol.

Observation and Action
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Field
     - Contract
   * - Observation
     - One ``cam_high`` RGB image, ordered ``cam_left_wrist`` and ``cam_right_wrist`` RGB images, and a 14D follower-joint state.
   * - Action
     - A 16-step chunk of absolute 14D dual-arm joint and gripper targets.
   * - Reward
     - Explicit sandwich success/failure/abort/timeout labels; unlabeled steps have zero reward.
   * - Prompt
     - ``make a sandwich`` with normalization assets selected by ``repo_id: aloha_sandwich``.

Installation
------------

.. include:: _setup_common.rst

Install the OpenPI stack and the lightweight dummy environment dependencies:

.. code-block:: bash

   bash requirements/install.sh embodied --model openpi --env maniskill_libero
   source .venv/bin/activate

What this does: it creates the RLinf environment, installs the OpenPI runtime used by ``openpi_rlinf``, and installs the verified OpenPI/ManiSkill-LIBERO dependency set used by the upstream RLT tests. The ALOHA adapter itself remains no-motion by default. Install the ROS and Interbotix dependencies for your robot-side process from the `official ALOHA instructions <https://github.com/tonyzhaozh/aloha>`_.

Prepare the Model and Data
--------------------------

Point the portable configs at your local assets. Keep ``repo_id``, action dimensions, and the 16-step horizon unchanged across stages.

.. code-block:: bash

   export OPENPI_MODEL_PATH=/path/to/openpi-base-checkpoint
   export ALOHA_DATASET_PATH=/path/to/lerobot-aloha-sandwich
   export ALOHA_NORM_STATS_PATH=/path/to/lerobot-aloha-sandwich/norm_stats.json
   export ALOHA_HDF5_DIR=/path/to/aloha-hdf5-episodes
   export RLINF_OUTPUT_DIR=/path/to/rlinf-results

The LeRobot dataset must contain the three named camera streams, 14D state/action data, and normalization statistics for ``aloha_sandwich``.

Run It
------

Train Stage 1:

.. code-block:: bash

   bash examples/sft/run_vla_sft.sh \
      aloha_sandwich_rlt_stage1_sft_openpi_pi05

What this does: it trains the new PyTorch ``openpi_rlinf`` π₀.₅ wrapper with ``action_horizon=16`` and the joint RLT reconstruction objective. Set ``ALOHA_STAGE1_CHECKPOINT`` to the resulting actor checkpoint directory.

Convert the offline episodes:

.. code-block:: bash

   export ALOHA_STAGE1_CHECKPOINT=/path/to/stage1/checkpoint/actor
   export ALOHA_REPLAY_BUFFER_PATH=/path/to/new/replay-buffer
   python toolkits/replay_buffer/convert_hdf5_to_rlinf_buffer.py

What this does: it extracts ``z_rl``, proprioception, and reference chunks with the frozen Stage-1 model, writes a new replay checkpoint, and validates its metadata, index, sample count, and trajectory files. The converter refuses to overwrite an existing output directory.

Train Stage 2 without robot or rollout workers:

.. code-block:: bash

   bash examples/embodiment/run_offline_rl.sh \
      aloha_sandwich_rlt_stage2_offline_ac_mlp

What this does: it shards complete trajectories across actor ranks and trains the RLT MLP actor-critic from the replay buffer. Resume a complete Stage-2 run with ``runner.resume_dir=/path/to/global_step_N``; never put the Stage-1 checkpoint in ``runner.ckpt_path``.

Serve a new-format Stage-1 policy on a separate deployment port:

.. code-block:: bash

   bash examples/serving/scripts/run_serve_pi05_aloha.sh \
      --config examples/serving/config/serve_pi05_aloha_sandwich.yaml \
      --smoke-test

The smoke test requires all three camera inputs, rejects non-finite or non-``(16, 14)`` actions, and validates Stage-2 feature shapes ``z_rl=(1, 2048)``, ``proprio=(1, 14)``, and ``ref_chunk=(1, 16, 14)``. Remove ``--smoke-test`` only after the metadata identifies the intended checkpoint, ``repo_id``, norm stats, horizon, and Git commit.

Legacy Checkpoint Isolation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Recreate the validated legacy service only in a separate worktree and virtual environment:

.. code-block:: bash

   export RLINF_INTEGRATION_REPO=/path/to/RLinf-integration
   git worktree add --detach ../RLinf-aloha-legacy \
      3eeb9265e4574ef0c046ecf0cfee892e8cc6e9aa
   cd ../RLinf-aloha-legacy
   bash requirements/install.sh embodied --model openpi \
      --env maniskill_libero --venv .venv-aloha-legacy
   source .venv-aloha-legacy/bin/activate
   python -m pip install --no-deps -r \
      "$RLINF_INTEGRATION_REPO/requirements/legacy/aloha_pi05_legacy_runtime.txt"
   export SERVING_PYTHON_BIN="$PWD/.venv-aloha-legacy/bin/python"
   bash examples/serving/scripts/run_serve_pi05_aloha.sh \
      --checkpoint-dir /path/to/legacy/checkpoint/actor \
      --repo-id pi05_sandwich_merged_all_0805 \
      --strict-load --smoke-test

What this does: it checks out the exact legacy source without moving either branch, installs the validated dependency snapshot into a dedicated environment, overrides the historical machine-specific checkpoint path, and requires a strict smoke test. The snapshot records exact installed distributions, so apply it with ``--no-deps`` after the normal installer; several upstream packages retain stale dependency bounds that conflict with the validated CUDA 12.8 overrides. The immutable source/configuration identities are recorded in ``examples/serving/config/legacy_pi05_aloha_runtime.yaml``. Never point ``SERVING_PYTHON_BIN`` at the integration environment.

.. warning::

   Old Stage-1 checkpoints use a different RLT decoder layout. Keep serving them from the preserved ``legacy/main-3eeb9265`` branch and its original environment. This runtime rejects ``checkpoint.format: legacy`` and never falls back to a non-strict load.

.. warning::

   The upstream integration currently exposes a deterministic no-motion ALOHA backend. Real action sending stays locked until a hardware backend implements joint limits, per-step delta limits, a watchdog, and emergency-stop handling. Do not override ``is_dummy: True`` or ``send_actions: False`` in ``realworld_aloha_sandwich.yaml`` yet.

Visualization and Results
-------------------------

Watch Stage-1 loss and Stage-2 actor/critic metrics in TensorBoard. For definitions, see :doc:`Training metrics <../../reference/metrics>`. Inspect ``conversion_manifest.json`` before training Stage 2; it records the source episodes and SHA256 identities for the Stage-1 weights and norm stats.

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Check
     - Expected result
   * - Stage-1 output
     - Actions have shape ``[batch, 16, 14]`` and finite values.
   * - Replay checkpoint
     - Metadata and index counts agree, every trajectory file exists, and every actor rank receives a non-empty shard.
   * - Policy smoke test
     - Metadata reports ``model_type: openpi_rlinf``, ``checkpoint_format: openpi_rlinf``, ``repo_id: aloha_sandwich``, and ``action_horizon: 16``; Stage-2 feature shapes are ``(1, 2048)``, ``(1, 14)``, and ``(1, 16, 14)``.

Read :doc:`Replay Buffer <../../concepts/replay_buffer>` for storage semantics and :doc:`Resume Training <../../guides/resume>` for full-run checkpoint recovery.
