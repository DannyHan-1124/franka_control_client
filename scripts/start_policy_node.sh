#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src:${WORKSPACE_ROOT}/lerobot/src:${PYTHONPATH:-}"

: "${ABPOLICY_CHECKPOINT:?Set ABPOLICY_CHECKPOINT to an ABPolicy pretrained_model directory}"
: "${ABPOLICY_DATASET_DIR:?Set ABPOLICY_DATASET_DIR to the joint-space training dataset}"

python -m franka_control_client.policy.pi05_policy_node \
    --checkpoint_path "$ABPOLICY_CHECKPOINT" \
    --dataset_path "$ABPOLICY_DATASET_DIR" \
    --device cuda \
    --policy_dtype bfloat16 \
    --fps 20 \
    --default_task "${ABPOLICY_TASK:-put red cylinder on blue cube}" \
    --direct_zmq_bind "${ABPOLICY_ZMQ_BIND:-tcp://0.0.0.0:40023}"
