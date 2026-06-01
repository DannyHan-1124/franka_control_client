from __future__ import annotations

import argparse
import time

import numpy as np
import pyzlc

from franka_control_client.core.latest_msg_subscriber import LatestMsgSubscriber


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Send fake Pi0.5 observations and verify action chunk shape."
    )
    parser.add_argument("--task", default="put red cylinder on green cube")
    parser.add_argument("--obs_topic", default="pi05/observation")
    parser.add_argument("--action_topic", default="pi05/action")
    parser.add_argument("--pyzlc_name", default="pi05_smoke_test")
    parser.add_argument("--pyzlc_host", default="192.168.0.109")
    parser.add_argument("--pyzlc_group_name", default="DroidGroup")
    parser.add_argument("--pyzlc_group_port", type=int, default=7730)
    parser.add_argument("--timeout_s", type=float, default=30.0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    pyzlc.init(
        args.pyzlc_name,
        args.pyzlc_host,
        group_name=args.pyzlc_group_name,
        group_port=args.pyzlc_group_port,
    )
    obs_pub = pyzlc.Publisher(args.obs_topic)
    action_sub = LatestMsgSubscriber(args.action_topic, wait_for_first_message=False)

    image = np.zeros((224, 224, 3), dtype=np.uint8)
    obs = {
        "observation.images.base_0_rgb": image,
        "observation.images.left_wrist_0_rgb": image,
        "observation.images.right_wrist_0_rgb": image,
        "observation.images.empty_camera_0": image,
        "observation.state": [0.48, 0.0, 0.14, 0.89, 0.41, 0.06, 0.07, 0.0],
        "task": args.task,
    }

    deadline = time.time() + args.timeout_s
    while time.time() < deadline:
        obs_pub.publish(obs)
        msg = action_sub.last_message
        if msg is not None:
            action = np.asarray(msg["action"], dtype=np.float32)
            print(f"Smoke test received action shape={action.shape}")
            if action.ndim != 2 or action.shape[-1] != 8:
                raise RuntimeError(f"Expected action chunk shape (T, 8), got {action.shape}")
            print("Smoke test PASSED.")
            pyzlc.shutdown()
            return
        time.sleep(0.1)

    pyzlc.shutdown()
    raise TimeoutError(f"No action received on {args.action_topic} within {args.timeout_s}s")


if __name__ == "__main__":
    main()
