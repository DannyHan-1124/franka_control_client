#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
STREAMING_MODE="${STREAMING_MODE:-chunk_replan}"
FASTER_INFER_TIME_SCHEDULE="${FASTER_INFER_TIME_SCHEDULE:-HAS}"
FASTER_ALPHA="${FASTER_ALPHA:-1.0}"
FASTER_U0="${FASTER_U0:-0.9}"
FASTER_DELAY_STEPS="${FASTER_DELAY_STEPS:-10}"
PHASE_FALLBACK_SCHEDULE="${PHASE_FALLBACK_SCHEDULE:-none}"
PHASE_FALLBACK_TRIGGER="${PHASE_FALLBACK_TRIGGER:-before_gripper_open}"

# kill old connection
# pgrep -af 'ssh.*(-L)'

python "${REPO_ROOT}/examples/pi05_policy_inference_franka.py" \
    --task "put red cylinder on blue cube and put green cylinder on yellow cube" \
    --stop_after_first_release \
    --robot_name FrankaPanda \
    --pyzlc_name policy_inference \
    --pyzlc_host 141.3.53.25 \
    --pyzlc_group_name robot_lab_robotiq_202 \
    --pyzlc_group_port 7725 \
    --policy_transport streaming_zmq \
    --policy_zmq_endpoint tcp://127.0.0.1:17726 \
    --streaming_mode "${STREAMING_MODE}" \
    --metrics_path "${REPO_ROOT}/logs/pi05_inference_metrics.jsonl" \
    --chunk_replan_steps 40 \
    --faster_infer_time_schedule "${FASTER_INFER_TIME_SCHEDULE}" \
    --faster_alpha "${FASTER_ALPHA}" \
    --faster_u0 "${FASTER_U0}" \
    --faster_delay_steps "${FASTER_DELAY_STEPS}" \
    --phase_fallback_schedule "${PHASE_FALLBACK_SCHEDULE}" \
    --phase_fallback_trigger "${PHASE_FALLBACK_TRIGGER}" \
    --static_camera static_cam \
    --wrist_camera wrist_cam
