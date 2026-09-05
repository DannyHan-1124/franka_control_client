#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src:${WORKSPACE_ROOT}/lerobot_f/src:${PYTHONPATH:-}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/$USER/matplotlib}"

export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${HOME}/.cache}"
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE" "$XDG_CACHE_HOME" "$MPLCONFIGDIR"

BASE_CHECKPOINT="${PI05_CHECKPOINT_PATH:-/data/zhuoyue/realrobot_ckpt/pi05_conveyor_cube_1500steps}"
: "${DSRL_CHECKPOINT:?Set DSRL_CHECKPOINT to an existing DSRL checkpoint for inference}"

python -m franka_control_client.policy.pi05_policy_DSRL_node \
    --weights "$BASE_CHECKPOINT" \
    --bind tcp://0.0.0.0:40023 \
    --output_dir "${DSRL_INFERENCE_OUTPUT_DIR:-/data/zhuoyue/franka_dsrl/inference_trajectories}" \
    --run_mode inference \
    --resume_checkpoint "$DSRL_CHECKPOINT" \
    --chunk_start_index "${DSRL_CHUNK_START_INDEX:-0}" \
    --chunk_trace_dir "${DSRL_CHUNK_TRACE_DIR:-/data/zhuoyue/franka_dsrl/chunk_traces}" \
    --dsrl_chunk_noise_viz_dir "${DSRL_NOISE_VIZ_DIR:-/data/zhuoyue/franka_dsrl/noise_visualizations}" \
    --rotation quat \
    --dsrl_state_dim 8 \
    --dsrl_num_images 2
