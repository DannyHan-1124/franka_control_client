#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src:${WORKSPACE_ROOT}/lerobot_f/src:${PYTHONPATH:-}"

# Keep Hugging Face on the user's authenticated cache by default. In particular,
# changing HF_HOME to /data hides the token created by `hf auth login` under
# ~/.cache/huggingface/token and makes gated PaliGemma downloads fail.
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${HOME}/.cache}"
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE" "$XDG_CACHE_HOME"

CHECKPOINT_PATH="${PI05_CHECKPOINT_PATH:-/data/zhuoyue/realrobot_ckpt/pi05_conveyor_cube_1500steps}"
DATASET_PATH="${PI05_DATASET_PATH:-/data/zhuoyue/realrobot_dataset/conveyor_cube}"

python -m franka_control_client.policy.pi05_policy_horizon_node \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --dataset_path "$DATASET_PATH" \
    --device cuda \
    --policy_dtype bfloat16 \
    --default_task "put red cube in bowl" \
    --direct_zmq_bind tcp://0.0.0.0:40023 \
    --call_vla_after_actions 20 \
    --chunk_start_index 2
