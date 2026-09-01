set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# kill old connection
# pgrep -af 'ssh.*(-L)'

python "${REPO_ROOT}/examples/pi05_policy_inference_franka.py" \
    --task "put red cylinder on red cube" \
    --stop_after_first_release \
    --pyzlc_name policy_inference \
    --pyzlc_host 141.3.53.25 \
    --pyzlc_group_name robot_lab_robotiq_202 \
    --pyzlc_group_port 7725 \
    --policy_transport zmq \
    --fps 20 \
    --chunk_replan_steps 35 \
    --policy_zmq_endpoint tcp://127.0.0.1:17725 \
    --static_camera static_cam \
    --wrist_camera wrist_cam

#   --metrics_path "${REPO_ROOT}/logs/pi05_puma_inference_metrics.jsonl" \
#   --stop_after_first_release \
#   --chunk_replan_steps 35 \