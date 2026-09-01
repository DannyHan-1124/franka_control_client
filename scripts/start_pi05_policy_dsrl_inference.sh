#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src:${WORKSPACE_ROOT}/lerobot_f/src:${PYTHONPATH:-}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/$USER/matplotlib}"

export HF_HOME="${HF_HOME:-/data/zhuoyue/cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/data/zhuoyue/cache/huggingface/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/data/zhuoyue/cache/huggingface/transformers}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/data/zhuoyue/cache}"
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE" "$XDG_CACHE_HOME" "$MPLCONFIGDIR"

BASE_CHECKPOINT="${PI05_CHECKPOINT_PATH:-/data/zhuoyue/realrobot_ckpt/pi05_conveyor_cube_1500steps}"
: "${DSRL_CHECKPOINT:?Set DSRL_CHECKPOINT to an existing DSRL checkpoint for inference}"

python -m franka_control_client.policy.pi05_policy_DSRL_node \
    --weights "$BASE_CHECKPOINT" \
    --bind tcp://0.0.0.0:40023 \
    --output_dir "${DSRL_INFERENCE_OUTPUT_DIR:-/scratch/$USER/franka_dsrl_inference_trajectories}" \
    --run_mode inference \
    --resume_checkpoint "$DSRL_CHECKPOINT" \
    --rotation quat \
    --dsrl_state_dim 8 \
    --dsrl_num_images 2
