#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src:${WORKSPACE_ROOT}/lerobot/src"

python -m franka_control_client.policy.pi05_policy_node \
    --checkpoint_path /hkfs/work/workspace/scratch/utphd-myspace/outputs/pi05_cylinder_full_10ksteps/checkpoints/010000/pretrained_model \
    --dataset_path /hkfs/work/workspace/scratch/utphd-myspace/datasets/cylinder_full \
    --device cuda \
    --policy_dtype bfloat16 \
    --obs_topic pi05/observation \
    --action_topic pi05/action \
    --fps 20 \
    --default_task "put green cylinder on blue cube" \
    --faster_infer_time_schedule const \
    --faster_alpha 1.0 \
    --faster_u0 0.9 \
    --streaming_zmq_bind tcp://0.0.0.0:40024
