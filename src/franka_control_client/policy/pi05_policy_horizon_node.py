#!/usr/bin/env python

"""PI0.5 action-chunk node with fixed-horizon replanning controls.

Unlike DOM_RL's simulator-facing client, this node talks to the real Franka
client.  ``chunk_start_index`` is therefore always a fixed integer; the
simulator-only ``auto`` delay compensation mode is intentionally unsupported.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any, Dict, Optional

import pyzlc
import torch

from lerobot.utils.random_utils import set_seed

from franka_control_client.policy.pi05_policy_node import Pi05NodeConfig, Pi05PolicyNode


@dataclass
class Pi05HorizonNodeConfig(Pi05NodeConfig):
    call_vla_after_actions: Optional[int]
    chunk_start_index: int


class Pi05PolicyHorizonNode(Pi05PolicyNode):
    """PI0.5 node that returns a fixed execution horizon and slices prefixes."""

    def _predict_action_msg(self, obs_msg: Dict[str, Any]) -> Dict[str, Any]:
        # The base implementation calls predict_action_chunk when available.
        message = super()._predict_action_msg(obs_msg)
        start = self.cfg.chunk_start_index
        actions = message["action"]
        raw_actions = message.get("action_raw")
        if start:
            actions = actions[start:]
            if raw_actions is not None:
                # action_raw has shape [B, T, D].
                raw_actions = [batch[start:] for batch in raw_actions]
        call_after = self.cfg.call_vla_after_actions
        if call_after is not None:
            actions = actions[:call_after]
            if raw_actions is not None:
                raw_actions = [batch[:call_after] for batch in raw_actions]
        if not actions:
            raise RuntimeError(
                f"--chunk_start_index={start} removed the entire predicted action chunk"
            )
        message.update(
            action=actions,
            shape=[len(actions), len(actions[0])],
            chunk_start_index=start,
            call_vla_after_actions=call_after,
        )
        if raw_actions is not None:
            message["action_raw"] = raw_actions
            message["raw_shape"] = [
                len(raw_actions),
                len(raw_actions[0]),
                len(raw_actions[0][0]),
            ]
        return message


def _parse_args() -> Pi05HorizonNodeConfig:
    parser = argparse.ArgumentParser(description="PI0.5 fixed-horizon policy node over pyzlc.")
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--dataset_path", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--policy_dtype", default="bfloat16")
    parser.add_argument("--obs_topic", default="pi05/observation")
    parser.add_argument("--action_topic", default="pi05/action")
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--default_task", default="")
    parser.add_argument("--pyzlc_name", default="pi05_policy_horizon_node")
    parser.add_argument("--pyzlc_host", default="0.0.0.0")
    parser.add_argument("--pyzlc_group_name", default="DroidGroup")
    parser.add_argument("--pyzlc_group_port", type=int, default=7730)
    parser.add_argument("--direct_zmq_bind", default=None)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--rtc_enabled", action="store_true")
    parser.add_argument("--rtc_execution_horizon", type=int, default=25)
    parser.add_argument("--rtc_max_guidance_weight", type=float, default=5.0)
    parser.add_argument(
        "--rtc_prefix_attention_schedule",
        default="exp",
        choices=("zeros", "ones", "linear", "exp"),
    )
    parser.add_argument("--rtc_debug", action="store_true")
    parser.add_argument(
        "--call_vla_after_actions",
        type=int,
        default=None,
        help="Replan once the active chunk has executed this many actions.",
    )
    parser.add_argument(
        "--chunk_start_index",
        type=int,
        default=0,
        help="Drop the first K predicted actions before publishing (no auto mode).",
    )
    args = parser.parse_args()
    if args.call_vla_after_actions is not None and args.call_vla_after_actions < 1:
        parser.error("--call_vla_after_actions must be >= 1")
    if args.chunk_start_index < 0:
        parser.error("--chunk_start_index must be >= 0")
    return Pi05HorizonNodeConfig(**vars(args))


def main() -> None:
    cfg = _parse_args()
    set_seed(cfg.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    node = Pi05PolicyHorizonNode(cfg)
    try:
        node.run()
    finally:
        pyzlc.shutdown()


if __name__ == "__main__":
    main()
