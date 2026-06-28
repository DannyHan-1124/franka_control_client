from __future__ import annotations

import argparse
import time

import pyzlc

from franka_control_client.core.latest_msg_subscriber import LatestMsgSubscriber
from franka_control_client.franka_robot.panda_arm import PandaArmState
from franka_control_client.robotiq_gripper.robotiq_gripper import (
    RemoteRobotiqGripper,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Continuously print the Franka joint positions and Robotiq jaw position."
        )
    )
    parser.add_argument("--robot_name", default="FrankaPanda")
    parser.add_argument("--pyzlc_name", default="robotiq_position_monitor")
    parser.add_argument("--pyzlc_host", default="141.3.53.25")
    parser.add_argument("--pyzlc_group_name", default="robot_lab_robotiq_202")
    parser.add_argument("--pyzlc_group_port", type=int, default=7725)
    parser.add_argument(
        "--rate_hz",
        type=float,
        default=20.0,
        help="Terminal update rate. The value is the latest received state.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.rate_hz <= 0:
        raise ValueError("--rate_hz must be greater than zero.")

    pyzlc.init(
        args.pyzlc_name,
        args.pyzlc_host,
        group_name=args.pyzlc_group_name,
        group_port=args.pyzlc_group_port,
    )

    # This monitor is deliberately read-only.
    arm_state_subscriber = LatestMsgSubscriber[PandaArmState](
        f"{args.robot_name}/franka_arm_state"
    )
    gripper = RemoteRobotiqGripper(args.robot_name, enable_publishers=False)
    period_s = 1.0 / args.rate_hz

    print(
        f"Monitoring {args.robot_name} joint and jaw states "
        "(jaw: 0.0=open, 1.0=closed). Press Ctrl-C to stop."
    )
    try:
        while True:
            arm_state = arm_state_subscriber.get_latest()
            gripper_state = gripper.current_state
            if arm_state is not None and gripper_state is not None:
                joints = [float(value) for value in arm_state["q"]]
                jaw_position = float(gripper_state["position"])
                raw_position = int(gripper_state["raw_position"])
                commanded = float(gripper_state["commanded_position"])
                joint_text = ", ".join(
                    f"q{index + 1}={value:+0.4f}"
                    for index, value in enumerate(joints)
                )
                print(
                    f"\r{joint_text} rad  "
                    f"jaw={jaw_position:0.4f}  "
                    f"raw={raw_position:3d}  commanded={commanded:0.4f}",
                    end="",
                    flush=True,
                )
            time.sleep(period_s)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        pyzlc.shutdown()


if __name__ == "__main__":
    main()
