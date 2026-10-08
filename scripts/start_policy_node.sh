#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src:${WORKSPACE_ROOT}/lerobot-abpolicy/src:${PYTHONPATH:-}"

python -m franka_control_client.policy.pi05_policy_node \
    --checkpoint_path /hkfs/work/workspace/scratch/utphd-myspace/outputs/pi05_conveyor_cube_abpolicy_2ksteps/checkpoints/001500/pretrained_model \
    --dataset_path /hkfs/work/workspace/scratch/utphd-myspace/datasets/conveyor_cube \
    --device cuda \
    --policy_dtype bfloat16 \
    --fps 20 \
    --default_task "put red cylinder on blue cube" \
    --direct_zmq_bind tcp://0.0.0.0:40023
