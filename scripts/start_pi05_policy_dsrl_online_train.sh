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
TRAJECTORY_DIR="${DSRL_TRAJECTORY_DIR:-/scratch/$USER/franka_dsrl_trajectories}"
CHECKPOINT_DIR="${DSRL_CHECKPOINT_DIR:-/scratch/$USER/franka_dsrl_checkpoints}"
RESUME_ARGS=()
if [[ -n "${DSRL_CHECKPOINT:-}" ]]; then
    RESUME_ARGS=(--resume_checkpoint "$DSRL_CHECKPOINT")
fi

python -m franka_control_client.policy.pi05_policy_DSRL_node \
    --weights "$BASE_CHECKPOINT" \
    --bind tcp://0.0.0.0:40023 \
    --output_dir "$TRAJECTORY_DIR" \
    --run_mode online_train \
    --save_checkpoint_dir "$CHECKPOINT_DIR" \
    --save_every_episodes 10 \
    --checkpoint_mode trainable \
    --rotation quat \
    --sac_gamma 0.99 \
    --sac_tau 0.005 \
    --sac_batch_size 64 \
    --sac_min_buffer_size 100 \
    --sac_replay_buffer_size 10000 \
    --sac_update_epochs 20 \
    --dsrl_state_dim 8 \
    --dsrl_action_noise_dim 32 \
    --dsrl_num_q_heads 10 \
    --dsrl_agg_q mean \
    --dsrl_num_images 2 \
    "${RESUME_ARGS[@]}"
