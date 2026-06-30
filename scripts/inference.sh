#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

# kill old connection
# pgrep -af 'ssh.*(-L)'

python "${REPO_ROOT}/examples/pi05_policy_inference_franka.py" \
    --task "put red cylinder on blue cube and put green cylinder on yellow cube" \
    --robot_name FrankaPanda \
    --pyzlc_name policy_inference \
    --pyzlc_host 141.3.53.25 \
    --pyzlc_group_name robot_lab_robotiq_202 \
    --pyzlc_group_port 7725 \
    --policy_transport streaming_zmq \
    --policy_zmq_endpoint tcp://127.0.0.1:17726 \
    --execution_horizon 50 \
    --faster_infer_time_schedule const \
    --faster_alpha 1.0 \
    --faster_u0 0.9 \
    --delay 0 \
    --early_stop_actions 0 \
    --phase_fallback_schedule none \
    --phase_fallback_trigger before_gripper_open \
    --static_camera static_cam \
    --wrist_camera wrist_cam

# --metrics_path "${REPO_ROOT}/logs/pi05_inference_metrics_overall.jsonl" \
# --gripper_open_confirm_steps 12 \
# --reclose_after_release_min_motion_m 0.08 \
# --stop_after_first_release \