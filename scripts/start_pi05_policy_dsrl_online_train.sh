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

BASE_CHECKPOINT="${PI05_CHECKPOINT_PATH:-/data/zhuoyue/realrobot_ckpt/0903/pi05_conveyor_cube_8_4_4_static_wrist_chunk20_lr5e05/5k}"
TRAJECTORY_DIR="${DSRL_TRAJECTORY_DIR:-/data/zhuoyue/franka_dsrl/trajectories}"
CHECKPOINT_DIR="${DSRL_CHECKPOINT_DIR:-/data/zhuoyue/franka_dsrl/checkpoints}"
TRACE_DIR="${DSRL_CHUNK_TRACE_DIR:-/data/zhuoyue/franka_dsrl/chunk_traces}"
NOISE_VIZ_DIR="${DSRL_NOISE_VIZ_DIR:-/data/zhuoyue/franka_dsrl/noise_visualizations}"
WANDB_PROJECT="${DSRL_WANDB_PROJECT:-franka_dsrl_online}"
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
    --save_every_episodes 5 \
    --checkpoint_mode trainable \
    --chunk_start_index "${DSRL_CHUNK_START_INDEX:-3}" \
    --chunk_trace_dir "$TRACE_DIR" \
    --dsrl_chunk_noise_viz_dir "$NOISE_VIZ_DIR" \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_run_name "${DSRL_WANDB_RUN_NAME:-franka_dsrl_online}" \
    --wandb_mode "${DSRL_WANDB_MODE:-online}" \
    --wandb_dir "${DSRL_WANDB_DIR:-/data/tmp/franka_dsrl_wandb}" \
    --rotation quat \
    --sac_gamma 0.99 \
    --sac_tau 0.005 \
    --sac_batch_size 64 \
    --sac_min_buffer_size 20 \
    --sac_replay_buffer_size 500 \
    --sac_update_epochs 20 \
    --dsrl_state_dim 8 \
    --dsrl_action_noise_dim 32 \
    --dsrl_action_magnitude 2.5 \
    --dsrl_hidden_dims 1024 1024 1024 \
    --dsrl_num_q_heads 2 \
    --dsrl_agg_q min \
    --dsrl_image_encoder_type temporal_conv \
    --dsrl_image_history_frames 3 \
    --dsrl_image_history_stride 3 \
    --dsrl_image_size 128 \
    --dsrl_image_latent_dim 128 \
    --dsrl_num_images 2 \
    "${RESUME_ARGS[@]}"


#sparse terminal reward no need for setting chunk discount
# 3 frames history span 0.3s
