#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src:${WORKSPACE_ROOT}/lerobot/src:${PYTHONPATH:-}"

: "${PUMA_CHECKPOINT:?Set PUMA_CHECKPOINT to the trained PI0.5-PUMA checkpoint}"
: "${PUMA_DATASET:?Set PUMA_DATASET to the matching read-only LeRobot dataset}"

python -m franka_control_client.policy.pi05_policy_node \
    --checkpoint_path "${PUMA_CHECKPOINT}" \
    --dataset_path "${PUMA_DATASET}" \
    --device cuda \
    --policy_dtype bfloat16 \
    --obs_topic pi05/observation \
    --action_topic pi05/action \
    --fps 20 \
    --default_task "put cylinder in moving cup" \
    --direct_zmq_bind tcp://0.0.0.0:40023
