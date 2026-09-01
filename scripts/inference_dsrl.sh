#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src:${WORKSPACE_ROOT}/lerobot_f/src:${PYTHONPATH:-}"

python "${REPO_ROOT}/examples/pi05_policy_inference_dsrl_franka.py" \
    --task "put red cube in bowl" \
    --pyzlc_name policy_inference_dsrl \
    --pyzlc_host 141.3.53.25 \
    --pyzlc_group_name robot_lab_robotiq_202 \
    --pyzlc_group_port 7725 \
    --policy_zmq_endpoint tcp://127.0.0.1:17725 \
    --metrics_path "${REPO_ROOT}/logs/pi05_dsrl_inference_metrics.jsonl" \
    --static_camera static_cam \
    --wrist_camera wrist_cam \
    --call_vla_after_actions 10 \
    --inference_latency 2
