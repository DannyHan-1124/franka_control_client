#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src:${WORKSPACE_ROOT}/lerobot/src"

# Keep Hugging Face's generated Arrow/dataset files off the quota-limited home
# filesystem. Respect explicit overrides when the caller already supplied them.
export HF_HOME="${HF_HOME:-${WORKSPACE_ROOT}/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
mkdir -p "${HF_DATASETS_CACHE}" "${HF_HUB_CACHE}"

python -m franka_control_client.policy.pi05_policy_node \
    --checkpoint_path /hkfs/work/workspace/scratch/utphd-myspace/outputs/pi05_cylinder_full_10ksteps/checkpoints/010000/pretrained_model \
    --dataset_path /hkfs/work/workspace/scratch/utphd-myspace/datasets/cylinder_full \
    --device cuda \
    --policy_dtype bfloat16 \
    --obs_topic pi05/observation \
    --action_topic pi05/action \
    --fps 20 \
    --default_task "put red cylinder on red cube" \
    --faster_infer_time_schedule HAS \
    --faster_alpha 0.6 \
    --faster_u0 0.9 \
    --direct_zmq_bind tcp://0.0.0.0:40024
