#!/usr/bin/env bash
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

# Launch the RLinf OpenPI Pi0.5 RLT Stage-1 SFT policy server (openpi-client WebSocket).
#
# Usage:
#   bash examples/serving/scripts/run_serve_pi05_aloha.sh \
#       --config examples/serving/config/serve_pi05_aloha_sandwich.yaml
#
# Optional env overrides:
#   SERVING_PYTHON_BIN   absolute interpreter (default: yushun-openpi conda python3.11)
#   SERVING_REPO_PATH    repository root (default: two levels above this script)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export REPO_PATH="${SERVING_REPO_PATH:-$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")}"

PYTHON_BIN="${SERVING_PYTHON_BIN:-/inspire/hdd/global_user/czxs253130583/yushun-home/miniforge3/envs/yushun-openpi/bin/python3.11}"

if [ ! -x "${PYTHON_BIN}" ]; then
  echo "ERROR: interpreter not found: ${PYTHON_BIN}" >&2
  exit 1
fi

# The deployment dependency set lives in ${REPO_PATH}/.venv (3.11-compiled site-packages).
# Explicit PYTHONPATH keeps this independent of any activated conda env / PYTHONHOME.
export PYTHONPATH="${REPO_PATH}:${REPO_PATH}/.venv/lib/python3.11/site-packages:${PYTHONPATH:-}"

exec "${PYTHON_BIN}" "${REPO_PATH}/examples/serving/scripts/serve_pi05_aloha.py" "$@"
