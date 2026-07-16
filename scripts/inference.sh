#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src:${WORKSPACE_ROOT}/lerobot/src:${PYTHONPATH:-}"

python "${REPO_ROOT}/examples/pi05_policy_inference_franka.py" \
    --task "${ABPOLICY_TASK:-put red cylinder on blue cube}" \
    --fps 20 \
    --control_hz 100 \
    --action_space cartesian \
    --abpolicy_enabled \
    --policy_transport zmq \
    --policy_zmq_endpoint "${ABPOLICY_ZMQ_ENDPOINT:-tcp://127.0.0.1:17725}" \
    --metrics_path "${REPO_ROOT}/logs/abpolicy_inference_metrics.jsonl" \
    --static_camera static_cam \
    --wrist_camera wrist_cam
