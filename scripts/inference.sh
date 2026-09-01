#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src:${WORKSPACE_ROOT}/lerobot/src:${PYTHONPATH:-}"

# Build or reuse the SSH tunnel to the lab RTX 5090 policy PC.
# Set SKIP_5090_TUNNEL=1 only when supplying the policy endpoint another way.
if [[ "${SKIP_5090_TUNNEL:-0}" != "1" ]]; then
    bash "${REPO_ROOT}/scripts/build_5090.sh"
fi
POLICY_ZMQ_ENDPOINT="${POLICY_ZMQ_ENDPOINT:-tcp://127.0.0.1:${LOCAL_POLICY_PORT:-17725}}"

python "${REPO_ROOT}/examples/pi05_policy_inference_franka.py" \
    --task "put red cube in bowl" \
    --stop_after_first_release \
    --pyzlc_name policy_inference \
    --pyzlc_host 141.3.53.25 \
    --pyzlc_group_name robot_lab_robotiq_202 \
    --pyzlc_group_port 7725 \
    --policy_transport zmq \
    --policy_zmq_endpoint "${POLICY_ZMQ_ENDPOINT}" \
    --metrics_path "${REPO_ROOT}/logs/pi05_inference_metrics_new.jsonl" \
    --static_camera static_cam \
    --wrist_camera wrist_cam \
    --rtc_enabled \
    --rtc_execution_horizon 10 \
    --rtc_delay_steps 7
