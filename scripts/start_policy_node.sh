#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src:${WORKSPACE_ROOT}/lerobot/src:${PYTHONPATH:-}"

export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${HOME}/.cache}"
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE" "$XDG_CACHE_HOME"

python -m franka_control_client.policy.pi05_policy_node \
    --checkpoint_path /data/zhuoyue/realrobot_ckpt/pi05_conveyor_cube_1500steps \
    --dataset_path /data/zhuoyue/realrobot_dataset/conveyor_cube \
    --device cuda \
    --policy_dtype bfloat16 \
    --obs_topic pi05/observation \
    --action_topic pi05/action \
    --fps 20 \
    --default_task "put red cube in bowl" \
    --direct_zmq_bind tcp://0.0.0.0:40023 \
    --rtc_enabled \
    --rtc_execution_horizon 25 \
    --rtc_max_guidance_weight 5.0 \
    --rtc_prefix_attention_schedule exp
