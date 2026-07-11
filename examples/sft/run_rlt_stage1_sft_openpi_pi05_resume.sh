#!/bin/bash

set -euo pipefail

# Resume RLT stage-1 OpenPI pi0.5 SFT.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 \
#   CFG_MICRO_BATCH_SIZE=1 \
#   CFG_GLOBAL_BATCH_SIZE=8 \
#   bash examples/sft/run_rlt_stage1_sft_openpi_pi05_resume.sh
#
# Optional overrides:
#   CFG_RESUME_DIR=/path/to/checkpoints/global_step_2000
#   CFG_LOG_PATH=/path/to/resume/output/root
#   CFG_GRADIENT_CHECKPOINTING=True
#   CFG_ACTOR_PLACEMENT=0-1
#   CFG_RESTART_RAY=0
#   CFG_RAY_PORT=6379

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export EMBODIED_PATH="${SCRIPT_DIR}"
export REPO_PATH="$(dirname "$(dirname "${SCRIPT_DIR}")")"
export SRC_FILE="${EMBODIED_PATH}/train_vla_sft.py"

if [ -z "${VIRTUAL_ENV:-}" ] && [ -f "${REPO_PATH}/.venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "${REPO_PATH}/.venv/bin/activate"
fi

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="${REPO_PATH}:${PYTHONPATH:-}"

CONFIG_NAME="${CFG_CONFIG_NAME:-rlt_stage1_sft_openpi_pi05}"
RESUME_DIR="${CFG_RESUME_DIR:-/inspire/qb-ilm/project/robot-reasoning/czxs253130583/yushun/results/rlt_stage1_sft_pi05_sandwich/rlt_stage1_sft_pi05_sandwich/checkpoints/global_step_2000}"
LOG_PATH="${CFG_LOG_PATH:-/inspire/qb-ilm/project/robot-reasoning/czxs253130583/yushun/results/rlt_stage1_sft_pi05_sandwich_resume}"
RAY_NODE_IP="${CFG_RAY_NODE_IP:-127.0.0.1}"
RAY_PORT="${CFG_RAY_PORT:-6379}"
RESTART_RAY="${CFG_RESTART_RAY:-1}"
GRADIENT_CHECKPOINTING="${CFG_GRADIENT_CHECKPOINTING:-True}"
ACTOR_PLACEMENT="${CFG_ACTOR_PLACEMENT:-0-0}"
CONFIG_DIR="${EMBODIED_PATH}/config"
TMP_CONFIG_DIR="$(mktemp -d /tmp/rlinf_sft_config.XXXXXX)"
cp -R "${CONFIG_DIR}/." "${TMP_CONFIG_DIR}/"
CONFIG_FILE="${TMP_CONFIG_DIR}/${CONFIG_NAME}.yaml"
if [ ! -f "${CONFIG_FILE}" ]; then
    echo "Config file not found: ${CONFIG_FILE}" >&2
    exit 1
fi
sed -i "s#^\([[:space:]]*actor,env,rollout:[[:space:]]*\).*#\1${ACTOR_PLACEMENT}#" "${CONFIG_FILE}"
CONFIG_DIR="${TMP_CONFIG_DIR}"

if [ ! -d "${RESUME_DIR}/actor" ]; then
    echo "Expected checkpoint actor directory not found: ${RESUME_DIR}/actor" >&2
    echo "Set CFG_RESUME_DIR to a checkpoint directory such as .../checkpoints/global_step_2000." >&2
    exit 1
fi

if [ "${RESTART_RAY}" = "1" ]; then
    ray stop --force
    ray start --head --node-ip-address="${RAY_NODE_IP}" --port="${RAY_PORT}"
fi
export RAY_ADDRESS="${RAY_NODE_IP}:${RAY_PORT}"

HYDRA_ARGS=(
    "+runner.resume_dir=${RESUME_DIR}"
    "runner.logger.log_path=${LOG_PATH}"
    "actor.fsdp_config.gradient_checkpointing=${GRADIENT_CHECKPOINTING}"
)

if [ -n "${CFG_MICRO_BATCH_SIZE:-}" ]; then
    HYDRA_ARGS+=("actor.micro_batch_size=${CFG_MICRO_BATCH_SIZE}")
fi

if [ -n "${CFG_GLOBAL_BATCH_SIZE:-}" ]; then
    HYDRA_ARGS+=("actor.global_batch_size=${CFG_GLOBAL_BATCH_SIZE}")
fi

if [ -n "${CFG_EXPERIMENT_NAME:-}" ]; then
    HYDRA_ARGS+=("runner.logger.experiment_name=${CFG_EXPERIMENT_NAME}")
fi

echo "Using Python at $(which python)"
echo "Using config ${CONFIG_NAME}"
echo "Using resume dir ${RESUME_DIR}"
echo "Using log path ${LOG_PATH}"

python "${SRC_FILE}" \
    --config-path "${CONFIG_DIR}" \
    --config-name "${CONFIG_NAME}" \
    "${HYDRA_ARGS[@]}" \
    "$@"
