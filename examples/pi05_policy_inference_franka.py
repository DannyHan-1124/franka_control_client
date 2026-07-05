from __future__ import annotations

import argparse
from typing import List

import pyzlc

from franka_control_client.camera.camera import CameraDevice
from franka_control_client.control_pair.cartesian_policy_panda_control_pair import (
    CartesianPolicyPandaRobotiqControlPair,
)
from franka_control_client.data_collection.irl_wrapper import (
    IRL_HardwareDataWrapper,
    ImageDataWrapper,
    PandaArmDataWrapper,
    RobotiqGripperDataWrapper,
)
from franka_control_client.franka_robot.panda_arm import RemotePandaArm
from franka_control_client.franka_robot.panda_robotiq import PandaRobotiq
from franka_control_client.policy_inference.pi05_policy_inference import (
    Pi05PolicyInference,
    Pi05PolicyInferenceConfig,
)
from franka_control_client.robotiq_gripper.robotiq_gripper import RemoteRobotiqGripper


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Franka Pi0.5 policy inference against a remote pyzlc policy node."
    )
    parser.add_argument("--task", required=True)
    parser.add_argument("--task_after_first_release", default=None)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--control_hz", type=float, default=100.0)
    parser.add_argument("--pyzlc_name", default="pi05_franka_client")
    parser.add_argument("--pyzlc_host", default="192.168.0.109")
    parser.add_argument("--pyzlc_group_name", default="DroidGroup")
    parser.add_argument("--pyzlc_group_port", type=int, default=7730)
    parser.add_argument("--policy_name", default="pi05")
    parser.add_argument("--obs_topic", default="pi05/observation")
    parser.add_argument("--action_topic", default="pi05/action")
    parser.add_argument("--policy_transport", choices=("pyzlc", "zmq"), default="pyzlc")
    parser.add_argument("--policy_zmq_endpoint", default=None)
    parser.add_argument("--policy_zmq_timeout_ms", type=int, default=30000)
    parser.add_argument("--chunk_replan_steps", type=int, default=50)
    parser.add_argument(
        "--rtc_enabled",
        action="store_true",
        help="Enable async Real-Time Chunking client execution. Requires --policy_transport zmq.",
    )
    parser.add_argument(
        "--rtc_execution_horizon",
        type=int,
        default=25,
        help="RTC minimum execution horizon s_min before starting the next async inference.",
    )
    parser.add_argument(
        "--rtc_delay_steps",
        type=int,
        default=0,
        help="RTC initial delay estimate d_init; observed request delays update this automatically.",
    )
    parser.add_argument(
        "--rtc_delay_buffer_size",
        type=int,
        default=8,
        help="Number of observed RTC inference delays used for the conservative max(Q) estimate.",
    )
    parser.add_argument("--gripper_open_confirm_steps", type=int, default=1)
    parser.add_argument("--stop_after_first_release", action="store_true")
    parser.add_argument("--stop_after_release_steps", type=int, default=0)
    parser.add_argument("--reclose_after_release_min_motion_m", type=float, default=0.08)
    parser.add_argument("--robot_name", default="FrankaPanda")
    parser.add_argument("--static_camera", default="static_cam")
    parser.add_argument("--wrist_camera", default="wrist_cam")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    pyzlc.init(
        args.pyzlc_name,
        args.pyzlc_host,
        group_name=args.pyzlc_group_name,
        group_port=args.pyzlc_group_port,
    )

    follower = PandaRobotiq(
        "PandaRobotiq",
        RemotePandaArm("FrankaPanda"),
        RemoteRobotiqGripper("FrankaPanda"),
    )
    control_pair = CartesianPolicyPandaRobotiqControlPair(
        follower.panda_arm,
        follower.robotiq_gripper,
        control_hz=args.control_hz,
    )

    data_collectors: List[IRL_HardwareDataWrapper] = [
        ImageDataWrapper(
            CameraDevice(args.static_camera, preview=False),
            hw_name="static_cam",
        ),
        ImageDataWrapper(
            CameraDevice(args.wrist_camera, preview=False),
            hw_name="wrist_cam",
        ),
        PandaArmDataWrapper(follower.panda_arm),
        RobotiqGripperDataWrapper(follower.robotiq_gripper),
    ]

    inference_cfg = Pi05PolicyInferenceConfig(
        policy_name=args.policy_name,
        task=args.task,
        task_after_first_release=args.task_after_first_release,
        fps=args.fps,
        obs_topic=args.obs_topic,
        action_topic=args.action_topic,
        policy_transport=args.policy_transport,
        policy_zmq_endpoint=args.policy_zmq_endpoint,
        policy_zmq_timeout_ms=args.policy_zmq_timeout_ms,
        chunk_replan_steps=args.chunk_replan_steps,
        rtc_enabled=args.rtc_enabled,
        rtc_execution_horizon=args.rtc_execution_horizon,
        rtc_delay_steps=args.rtc_delay_steps,
        rtc_delay_buffer_size=args.rtc_delay_buffer_size,
        gripper_open_confirm_steps=args.gripper_open_confirm_steps,
        stop_after_first_release=args.stop_after_first_release,
        stop_after_release_steps=args.stop_after_release_steps,
        reclose_after_release_min_motion_m=args.reclose_after_release_min_motion_m,
    )
    inference_manager = Pi05PolicyInference(
        data_collectors=data_collectors,
        control_pair=control_pair,
        cfg=inference_cfg,
    )

    try:
        inference_manager.run()
    finally:
        pyzlc.shutdown()


if __name__ == "__main__":
    main()
