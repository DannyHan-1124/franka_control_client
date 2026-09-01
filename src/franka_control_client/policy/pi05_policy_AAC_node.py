#!/usr/bin/env python

"""Adaptive Action Chunking PI0.5 node for the real Franka client."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pyzlc
import torch

from lerobot.utils.random_utils import set_seed

from franka_control_client.policy.pi05_policy_node import Pi05NodeConfig, Pi05PolicyNode


WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
AAC_ROOT = WORKSPACE_ROOT / "AAC-Pi05"
if str(AAC_ROOT) not in sys.path:
    sys.path.insert(0, str(AAC_ROOT))

from aac_pi05 import AACConfig, AACPI05  # noqa: E402


@dataclass
class Pi05AACNodeConfig(Pi05NodeConfig):
    aac_num_samples: int
    aac_entropy_method: str
    aac_move_threshold: float
    aac_motion_action_mode: str
    aac_max_horizon: Optional[int]
    aac_chunk_selector: str
    aac_backward_beta: float
    aac_entropy_log: Optional[str]
    aac_log_entropy_values: bool
    chunk_start_index: int


class Pi05PolicyAACNode(Pi05PolicyNode):
    """PI0.5 multi-sample inference with an entropy-selected dynamic horizon."""

    def __init__(self, cfg: Pi05AACNodeConfig) -> None:
        if cfg.rtc_enabled:
            raise ValueError("AAC and RTC cannot be enabled together")
        self._aac_episode_id: Any = object()
        self._aac_inference_id = 0
        super().__init__(cfg)
        self.aac = AACPI05(
            self.policy,
            postprocessor=self.postprocessor,
            config=AACConfig(
                num_samples=cfg.aac_num_samples,
                entropy_method=cfg.aac_entropy_method,
                move_threshold=cfg.aac_move_threshold,
                motion_action_mode=cfg.aac_motion_action_mode,
                max_horizon=cfg.aac_max_horizon,
                chunk_selector=cfg.aac_chunk_selector,
                backward_beta=cfg.aac_backward_beta,
            ),
            action_layout="franka_quat8",
        )
        self._prepare_entropy_log()

    def _prepare_entropy_log(self) -> None:
        if not self.cfg.aac_entropy_log:
            return
        path = Path(self.cfg.aac_entropy_log)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            with path.open("w", newline="", encoding="utf-8") as stream:
                csv.writer(stream).writerow(
                    [
                        "timestamp", "episode_id", "inference_id", "horizon",
                        "sample_id", "K_entropy", "K_motion", "K_raw",
                        "horizon_source", "horizon_capped", "entropy_mean",
                    ]
                )

    def _maybe_reset_episode(self, obs_msg: Dict[str, Any]) -> None:
        episode_id = obs_msg.get("episode_id", 0)
        if episode_id != self._aac_episode_id:
            self.aac.reset()
            self._aac_episode_id = episode_id
            self._aac_inference_id = 0

    def _log_prediction(self, prediction: Any) -> None:
        total = np.asarray(prediction.entropy["total"], dtype=float)
        if self.cfg.aac_log_entropy_values:
            message = (
                f"AAC horizon={prediction.horizon}, sample={prediction.sample_id}, "
                f"total_entropy={total.tolist()}"
            )
            if self.cfg.direct_zmq_bind:
                print(message, flush=True)
            else:
                pyzlc.info(message)
        if self.cfg.aac_entropy_log:
            with Path(self.cfg.aac_entropy_log).open("a", newline="", encoding="utf-8") as stream:
                csv.writer(stream).writerow(
                    [
                        time.time(), self._aac_episode_id, self._aac_inference_id,
                        prediction.horizon, prediction.sample_id,
                        prediction.entropy.get("K_entropy"),
                        prediction.entropy.get("K_motion"),
                        prediction.entropy.get("K_raw"),
                        prediction.entropy.get("horizon_source"),
                        prediction.entropy.get("horizon_capped"), float(total.mean()),
                    ]
                )
        self._aac_inference_id += 1

    def _predict_action_msg(self, obs_msg: Dict[str, Any]) -> Dict[str, Any]:
        self._maybe_reset_episode(obs_msg)
        observation = self.preprocessor(self._build_observation(obs_msg))
        current_state = np.asarray(obs_msg["observation.state"], dtype=np.float32).reshape(-1)
        prediction = self.aac.predict(
            observation,
            current_state=current_state,
            rotation_format="quat_xyzw",
        )
        self._log_prediction(prediction)

        start = self.cfg.chunk_start_index
        selected = prediction.actions[:, start:, :]
        if selected.shape[1] == 0:
            raise RuntimeError(
                f"--chunk_start_index={start} removed AAC horizon={prediction.horizon}"
            )
        actions = selected[0].detach().float().cpu().numpy()
        return {
            "timestamp": time.time(),
            "action": actions.tolist(),
            "shape": list(actions.shape),
            "aac_horizon": int(prediction.horizon),
            "execution_horizon": int(actions.shape[0]),
            "aac_sample_id": int(prediction.sample_id),
            "chunk_start_index": start,
        }


def _parse_args() -> Pi05AACNodeConfig:
    parser = argparse.ArgumentParser(description="PI0.5 AAC policy node for Franka.")
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--dataset_path", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--policy_dtype", default="bfloat16")
    parser.add_argument("--obs_topic", default="pi05/observation")
    parser.add_argument("--action_topic", default="pi05/action")
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--default_task", default="")
    parser.add_argument("--pyzlc_name", default="pi05_policy_AAC_node")
    parser.add_argument("--pyzlc_host", default="0.0.0.0")
    parser.add_argument("--pyzlc_group_name", default="DroidGroup")
    parser.add_argument("--pyzlc_group_port", type=int, default=7730)
    parser.add_argument("--direct_zmq_bind", default=None)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--rtc_enabled", action="store_true")
    parser.add_argument("--rtc_execution_horizon", type=int, default=25)
    parser.add_argument("--rtc_max_guidance_weight", type=float, default=5.0)
    parser.add_argument(
        "--rtc_prefix_attention_schedule", default="exp",
        choices=("zeros", "ones", "linear", "exp"),
    )
    parser.add_argument("--rtc_debug", action="store_true")
    parser.add_argument("--aac_num_samples", type=int, default=20)
    parser.add_argument(
        "--aac_entropy_method", default="gaussian_bernoulli",
        choices=("gaussian_bernoulli", "gaussian_only", "variance", "separate", "binning"),
    )
    parser.add_argument("--aac_move_threshold", type=float, default=3.0)
    parser.add_argument(
        "--aac_motion_action_mode", default="absolute_to_delta",
        choices=("absolute_to_delta", "disabled"),
    )
    parser.add_argument("--aac_max_horizon", type=int, default=None)
    parser.add_argument(
        "--aac_chunk_selector", default="backward", choices=("first", "mean", "backward")
    )
    parser.add_argument("--aac_backward_beta", type=float, default=0.99)
    parser.add_argument("--aac_entropy_log", default=None)
    parser.add_argument("--aac_log_entropy_values", action="store_true")
    parser.add_argument("--chunk_start_index", type=int, default=0)
    args = parser.parse_args()
    if args.rtc_enabled:
        parser.error("AAC and RTC cannot be enabled together")
    if args.chunk_start_index < 0:
        parser.error("--chunk_start_index must be >= 0")
    return Pi05AACNodeConfig(**vars(args))


def main() -> None:
    cfg = _parse_args()
    set_seed(cfg.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    node = Pi05PolicyAACNode(cfg)
    try:
        node.run()
    finally:
        pyzlc.shutdown()


if __name__ == "__main__":
    main()
