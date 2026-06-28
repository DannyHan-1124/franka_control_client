#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

# kill old connection
# pgrep -af 'ssh.*(-L)'

python "${REPO_ROOT}/examples/pi05_policy_inference_franka.py" \
    --task "put red cylinder on red cube" \
    --stop_after_first_release \
    --robot_name FrankaPanda \
    --pyzlc_name policy_inference \
    --pyzlc_host 141.3.53.25 \
    --pyzlc_group_name robot_lab_robotiq_202 \
    --pyzlc_group_port 7725 \
    --policy_transport streaming_zmq \
    --policy_zmq_endpoint tcp://127.0.0.1:17726 \
    --metrics_path "${REPO_ROOT}/logs/pi05_inference_metrics_formal.jsonl" \
    --execution_horizon 40 \
    --faster_infer_time_schedule HAS \
    --faster_alpha 0.6 \
    --faster_u0 0.9 \
    --delay 10 \
    --early_stop_actions 0 \
    --phase_fallback_schedule none \
    --phase_fallback_trigger before_gripper_open \
    --static_camera static_cam \
    --wrist_camera wrist_cam
