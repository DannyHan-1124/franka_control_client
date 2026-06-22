#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

# kill old connection
# pgrep -af 'ssh.*(-L)'

python "${REPO_ROOT}/examples/pi05_policy_inference_franka.py" \
    --task "put green cylinder on blue cube" \
    --stop_after_first_release \
    --robot_name FrankaPanda \
    --pyzlc_name policy_inference \
    --pyzlc_host 141.3.53.25 \
    --pyzlc_group_name robot_lab_robotiq_202 \
    --pyzlc_group_port 7725 \
    --policy_transport streaming_zmq \
    --policy_zmq_endpoint tcp://127.0.0.1:17726 \
    --streaming_mode continuous \
    --continuous_min_execute_steps 20 \
    --metrics_path "${REPO_ROOT}/logs/pi05_inference_metrics_c.jsonl" \
    --static_camera static_cam \
    --wrist_camera wrist_cam
