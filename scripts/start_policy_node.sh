#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src:${WORKSPACE_ROOT}/lerobot-mpq/src:${PYTHONPATH:-}"

CHECKPOINT_PATH="${PI05_CHECKPOINT_PATH:-${WORKSPACE_ROOT}/outputs/pi05_cylinder_full_10ksteps/checkpoints/last/pretrained_model}"

python -m franka_control_client.policy.pi05_policy_node \
    --checkpoint_path "${CHECKPOINT_PATH}" \
    --dataset_path "${WORKSPACE_ROOT}/datasets/cylinder_full" \
    --device cuda \
    --policy_dtype bfloat16 \
    --fps 20 \
    --default_task "put red cylinder on blue cube" \
    --direct_zmq_bind tcp://0.0.0.0:40023
