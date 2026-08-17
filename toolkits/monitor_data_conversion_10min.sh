#!/usr/bin/env bash

# Observe both currently running HDF5 -> LeRobot v2.1 conversions for a
# bounded period. This script never stops, kills, or modifies either process.
#
# Usage:
#   bash toolkits/monitor_data_conversion_10min.sh [duration_seconds] [interval_seconds]
#
# Example:
#   nohup bash toolkits/monitor_data_conversion_10min.sh 600 60 \
#     > /inspire/hdd/global_user/czxs253130583/fangchuan/log/data_conversion/conversion_monitor_10min.log 2>&1 &

set -u

DURATION_SECONDS="${1:-600}"
INTERVAL_SECONDS="${2:-60}"
FORGE_PID="${FORGE_PID:-2440876}"
OFFICIAL_PID="${OFFICIAL_PID:-1878057}"

FORGE_OUT="/inspire/hdd/global_user/czxs253130583/data/lerobot-data/sandwich_merged_all_0805_v21_forge"
OFFICIAL_OUT="/inspire/hdd/global_user/czxs253130583/data/lerobot-data/sandwich_merged_all_0805_v21"

if ! [[ "$DURATION_SECONDS" =~ ^[0-9]+$ && "$INTERVAL_SECONDS" =~ ^[1-9][0-9]*$ ]]; then
    echo "Usage: $0 [duration_seconds>=0] [interval_seconds>=1]" >&2
    exit 2
fi

snapshot() {
    local label="$1"
    local pid="$2"
    local output="$3"
    echo "[$(date '+%F %T')] ${label} pid=${pid}"
    ps -p "$pid" -o pid=,etime=,pcpu=,pmem=,stat=,args= 2>/dev/null || echo "  process_not_running"
    if [ -d "$output" ]; then
        echo "  output_size=$(du -sh "$output" 2>/dev/null | awk '{print $1}')"
        echo "  parquet_count=$(find "$output/data" -type f -name '*.parquet' 2>/dev/null | wc -l | tr -d ' ')"
        if [ -f "$output/meta/info.json" ]; then
            python - "$output/meta/info.json" <<'PY'
import json
import sys

try:
    info = json.load(open(sys.argv[1]))
    print(
        "  metadata="
        f"version={info.get('codebase_version')} "
        f"episodes={info.get('total_episodes')} "
        f"frames={info.get('total_frames')}"
    )
except Exception as exc:
    print(f"  metadata_read_error={exc}")
PY
        else
            echo "  metadata=not_finalized"
        fi
    else
        echo "  output=not_created"
    fi
}

START_EPOCH="$(date +%s)"
END_EPOCH=$((START_EPOCH + DURATION_SECONDS))
while :; do
    NOW="$(date +%s)"
    ELAPSED=$((NOW - START_EPOCH))
    echo "===== conversion monitor elapsed=${ELAPSED}s/${DURATION_SECONDS}s ====="
    snapshot "forge_v21" "$FORGE_PID" "$FORGE_OUT"
    snapshot "official_v21" "$OFFICIAL_PID" "$OFFICIAL_OUT"
    if [ "$NOW" -ge "$END_EPOCH" ]; then
        break
    fi
    REMAINING=$((END_EPOCH - NOW))
    sleep "$((REMAINING < INTERVAL_SECONDS ? REMAINING : INTERVAL_SECONDS))"
done

echo "===== conversion monitor finished after ${DURATION_SECONDS}s ====="
