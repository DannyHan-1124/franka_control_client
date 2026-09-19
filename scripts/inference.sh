#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src:${WORKSPACE_ROOT}/mpq:${PYTHONPATH:-}"

MPQ_LIBRARY="${MPQ_LIBRARY:-${WORKSPACE_ROOT}/mpq/libraries/cylinder_full_h50_yaw_k256_g5.pt}"
# cylinder_full_yaw_k256_g5.pt
# cylinder_full_h50_yaw_k256_g5.pt
if [[ ! -f "${MPQ_LIBRARY}" ]]; then
    echo "MPQ library not found: ${MPQ_LIBRARY}" >&2
    echo "Build it with ${WORKSPACE_ROOT}/mpq/scripts/build_cylinder_full_library.sh" >&2
    exit 1
fi

# kill old connection
# pgrep -af 'ssh.*(-L)'

python "${REPO_ROOT}/examples/pi05_policy_inference_franka.py" \
    --task "put green cylinder on blue cube" \
    --stop_after_first_release \
    --fps 20 \
    --control_hz 100 \
    --robot_name FrankaPanda \
    --pyzlc_name policy_inference \
    --pyzlc_host 141.3.53.25 \
    --pyzlc_group_name robot_lab_robotiq_202 \
    --pyzlc_group_port 7725 \
    --policy_transport zmq \
    --policy_zmq_endpoint tcp://127.0.0.1:17725 \
    --metrics_path "${REPO_ROOT}/logs/pi05_mpq_inference_metrics.jsonl" \
    --chunk_replan_steps 50 \
    --mpq_library "${MPQ_LIBRARY}" \
    --mpq_delta "${MPQ_DELTA:-0.25}" \
    --mpq_metric_horizon 20 \
    --mpq_device "${MPQ_DEVICE:-cpu}" \
    --static_camera static_cam \
    --wrist_camera wrist_cam

#   --mpq_library "${MPQ_LIBRARY}" \