#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src:${WORKSPACE_ROOT}/lerobot/src:${PYTHONPATH:-}"

export HF_HOME=/scratch/$USER/hf_home
export HF_DATASETS_CACHE=/scratch/$USER/hf_datasets
export TRANSFORMERS_CACHE=/scratch/$USER/hf_transformers
export XDG_CACHE_HOME=/scratch/$USER/cache
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE" "$XDG_CACHE_HOME"

python -m franka_control_client.policy.pi05_policy_node \
    --checkpoint_path /hkfs/work/workspace/scratch/utphd-myspace/outputs/pi05_moving_cup_10ksteps_new/checkpoints/010000/pretrained_model \
    --dataset_path /hkfs/work/workspace/scratch/utphd-myspace/datasets/moving_cup \
    --device cuda \
    --policy_dtype bfloat16 \
    --obs_topic pi05/observation \
    --action_topic pi05/action \
    --fps 20 \
    --default_task "put cylinder in moving cup" \
    --direct_zmq_bind tcp://0.0.0.0:40023 \
    --rtc_enabled \
    --rtc_execution_horizon 25 \
    --rtc_max_guidance_weight 5.0 \
    --rtc_prefix_attention_schedule exp
