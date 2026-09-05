#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src:${WORKSPACE_ROOT}/lerobot_f/src:${PYTHONPATH:-}"

export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${HOME}/.cache}"
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE" "$XDG_CACHE_HOME"

CHECKPOINT_PATH="${PI05_CHECKPOINT_PATH:-/data/zhuoyue/realrobot_ckpt/0903/pi05_conveyor_cube_8_4_4_static_wrist_chunk20_lr5e05/5k}"
DATASET_PATH="${PI05_DATASET_PATH:-/data/zhuoyue/realrobot_dataset/conveyor_cube}"

python -m franka_control_client.policy.pi05_policy_AAC_node \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --dataset_path "$DATASET_PATH" \
    --device cuda \
    --policy_dtype bfloat16 \
    --default_task "put red cube in bowl" \
    --direct_zmq_bind tcp://0.0.0.0:40023 \
    --aac_num_samples 2 \
    --aac_entropy_method gaussian_bernoulli \
    --aac_move_threshold 0.4 \
    --aac_motion_action_mode absolute_to_delta \
    --aac_max_horizon 20 \
    --aac_chunk_selector backward \
    --aac_backward_beta 0.99 \
    --chunk_start_index 6 \
    --aac_entropy_log "${AAC_ENTROPY_LOG:-/data/zhuoyue/logs/franka_aac_pi05_entropy.csv}"


# move threshold cant set to 0, or the motion will be too sudden
# visualize the chunk
# constraint motion