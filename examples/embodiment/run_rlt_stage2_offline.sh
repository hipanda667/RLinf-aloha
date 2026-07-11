#!/usr/bin/env bash
set -euo pipefail

# 假设脚本位于：
# <repo>/examples/embodiment/run_rlt_stage2_offline.sh
EMBODIED_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_PATH="$(cd "${EMBODIED_PATH}/../.." && pwd)"

export EMBODIED_PATH
export REPO_PATH
export PYTHONPATH="${REPO_PATH}:${PYTHONPATH:-}"
export ROBOT_PLATFORM="${ROBOT_PLATFORM:-ALOHA}"

cd "${REPO_PATH}"

# 自动激活当前仓库的虚拟环境。
if [[ -f "${REPO_PATH}/.venv/bin/activate" ]]; then
    source "${REPO_PATH}/.venv/bin/activate"
fi

CONFIG_NAME="${CONFIG_NAME:-rlt_stage2_offline_ac_mlp}"
MAX_STEPS="${MAX_STEPS:-10000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-500}"

echo "REPO_PATH=${REPO_PATH}"
echo "EMBODIED_PATH=${EMBODIED_PATH}"
echo "ROBOT_PLATFORM=${ROBOT_PLATFORM}"
echo "CONFIG_NAME=${CONFIG_NAME}"
echo "MAX_STEPS=${MAX_STEPS}"
echo "SAVE_INTERVAL=${SAVE_INTERVAL}"
echo "Python: $(command -v python)"

python "${EMBODIED_PATH}/train_rlt_stage2_offline.py" \
    --config-name "${CONFIG_NAME}" \
    runner.max_steps="${MAX_STEPS}" \
    runner.save_interval="${SAVE_INTERVAL}" \
    "$@"