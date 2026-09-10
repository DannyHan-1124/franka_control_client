#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src:${WORKSPACE_ROOT}/lerobot-abpolicy/src:${PYTHONPATH:-}"

python -m franka_control_client.policy.pi05_policy_node \
    --checkpoint_path /hkfs/work/workspace/scratch/utphd-myspace/outputs/pi05_moving_cup_mixed_abpolicy_3ksteps/checkpoints/last/pretrained_model \
    --dataset_path /hkfs/work/workspace/scratch/utphd-myspace/datasets/moving_cup_mixed \
    --device cuda \
    --policy_dtype bfloat16 \
    --fps 20 \
    --default_task "put red cylinder on blue cube" \
    --direct_zmq_bind tcp://0.0.0.0:40023
