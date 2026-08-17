#!/bin/bash

# Start RLT stage-1 OpenPI pi0.5 SFT from the base model.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0,1 \
#   CFG_ACTOR_PLACEMENT=0-1 \
#   CFG_MICRO_BATCH_SIZE=1 \
#   CFG_GLOBAL_BATCH_SIZE=8 \
#   CFG_LOG_PATH=/path/to/output/root \
#   bash examples/sft/run_rlt_stage1_sft_openpi_pi05.sh
#
# Optional overrides:
#   CFG_CONFIG_NAME=rlt_stage1_sft_openpi_pi05
#   CFG_EXPERIMENT_NAME=rlt_stage1_sft_pi05_sandwich_h200_2gpu
#   CFG_GRADIENT_CHECKPOINTING=True
#   CFG_RESTART_RAY=0
#   CFG_RAY_PORT=6379

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export EMBODIED_PATH="${SCRIPT_DIR}"
export REPO_PATH="$(dirname "$(dirname "${SCRIPT_DIR}")")"
export SRC_FILE="${EMBODIED_PATH}/train_vla_sft.py"

# Keep all training-side temporary files, caches, Ray state, and logger data on
# the HDD work area.  Do not add these variables to env.sh: this script is the
# isolated entry point for the RLT training workflow.
export HDD_WORK="/inspire/hdd/global_user/czxs253130583/fangchuan"
export XDG_CACHE_HOME="${HDD_WORK}/.cache"
export TMPDIR="${HDD_WORK}/tmp"
export TEMP="${TMPDIR}"
export TMP="${TMPDIR}"
# Ray creates Unix sockets below this directory. The socket path is exposed
# through a short /tmp symlink because AF_UNIX paths are limited to 107 bytes;
# the symlink target keeps the Ray state on the HDD.
export RAY_TMPDIR="${HDD_WORK}/tmp/r"
export RAY_TMPDIR_LINK="/tmp/rlinf-ray"
export TORCH_HOME="${XDG_CACHE_HOME}/torch"
export TRITON_CACHE_DIR="${XDG_CACHE_HOME}/triton"
export CUDA_CACHE_PATH="${XDG_CACHE_HOME}/cuda"
export HF_HOME="${XDG_CACHE_HOME}/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export WANDB_DIR="${HDD_WORK}/log/wandb"
mkdir -p \
    "${TMPDIR}" \
    "${RAY_TMPDIR}" \
    "${TORCH_HOME}" \
    "${TRITON_CACHE_DIR}" \
    "${CUDA_CACHE_PATH}" \
    "${HF_HUB_CACHE}" \
    "${HF_DATASETS_CACHE}" \
    "${TRANSFORMERS_CACHE}" \
    "${WANDB_DIR}"

# The training dependency set lives in ${REPO_PATH}/.venv (3.11-compiled
# wheels), but the uv interpreter it symlinks to is not mounted in every
# container. The yushun-openpi conda env provides a CPython 3.11 that can
# load those wheels, so it is used as the interpreter while site-packages
# stay in the venv.
PYTHON_BIN="/inspire/hdd/global_user/czxs253130583/yushun-home/miniforge3/envs/yushun-openpi/bin/python3.11"
if [ ! -x "${PYTHON_BIN}" ]; then
    echo "python unavailable: ${PYTHON_BIN}" >&2
    exit 1
fi
# Prevent a conda/base PYTHONHOME from overriding the interpreter's stdlib.
unset PYTHONHOME

set -euo pipefail

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="${REPO_PATH}:${REPO_PATH}/.venv/lib/python3.11/site-packages:${REPO_PATH}/.venv/libero:${PYTHONPATH:-}"
if [ -e "${RAY_TMPDIR_LINK}" ] && [ ! -L "${RAY_TMPDIR_LINK}" ]; then
    echo "Ray temporary path exists and is not a symlink: ${RAY_TMPDIR_LINK}" >&2
    exit 1
fi
ln -sfnT "${RAY_TMPDIR}" "${RAY_TMPDIR_LINK}"

CONFIG_NAME="${CFG_CONFIG_NAME:-rlt_stage1_sft_openpi_pi05}"
LOG_PATH="${CFG_LOG_PATH:-${REPO_PATH}/results/rlt_stage1_sft_pi05_sandwich_h200_2gpu}"
RAY_NODE_IP="${CFG_RAY_NODE_IP:-127.0.0.1}"
RAY_PORT="${CFG_RAY_PORT:-6379}"
RESTART_RAY="${CFG_RESTART_RAY:-1}"
GRADIENT_CHECKPOINTING="${CFG_GRADIENT_CHECKPOINTING:-True}"
ACTOR_PLACEMENT="${CFG_ACTOR_PLACEMENT:-0-0}"
CONFIG_DIR="${EMBODIED_PATH}/config"
TMP_CONFIG_DIR="$(mktemp -d "${TMPDIR}/rlinf_sft_config.XXXXXX")"
cp -R "${CONFIG_DIR}/." "${TMP_CONFIG_DIR}/"
CONFIG_FILE="${TMP_CONFIG_DIR}/${CONFIG_NAME}.yaml"
if [ ! -f "${CONFIG_FILE}" ]; then
    echo "Config file not found: ${CONFIG_FILE}" >&2
    exit 1
fi
sed -i "s#^\([[:space:]]*actor,env,rollout:[[:space:]]*\).*#\1${ACTOR_PLACEMENT}#" "${CONFIG_FILE}"
CONFIG_DIR="${TMP_CONFIG_DIR}"

RAY_CMD() {
    RAY_ARGV="$*" "${PYTHON_BIN}" - <<'PY'
import os, runpy, sys
sys.argv = ["ray"] + os.environ["RAY_ARGV"].split()
runpy.run_module("ray.scripts.scripts", run_name="__main__")
PY
}

if [ "${RESTART_RAY}" = "1" ]; then
    RAY_CMD stop --force
    RAY_CMD start --head --temp-dir="${RAY_TMPDIR_LINK}" --node-ip-address="${RAY_NODE_IP}" --port="${RAY_PORT}"
fi
export RAY_ADDRESS="${RAY_NODE_IP}:${RAY_PORT}"

HYDRA_ARGS=(
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

echo "Using Python at ${PYTHON_BIN}"
echo "Using config ${CONFIG_NAME}"
echo "Using log path ${LOG_PATH}"

"${PYTHON_BIN}" "${SRC_FILE}" \
    --config-path "${CONFIG_DIR}" \
    --config-name "${CONFIG_NAME}" \
    "${HYDRA_ARGS[@]}" \
    "$@"
