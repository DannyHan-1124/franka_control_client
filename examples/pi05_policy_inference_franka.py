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
    parser.add_argument("--task", required=True, help="Language command sent to the policy.")
    parser.add_argument("--fps", type=int, default=20, help="High-level policy action application rate.")
    parser.add_argument("--control_hz", type=float, default=100.0, help="Low-level Cartesian control loop rate.")
    parser.add_argument("--robot_name", default="FrankaPanda", help="pyzlc robot namespace for arm and gripper topics.")
    parser.add_argument("--pyzlc_name", default="pi05_franka_client")
    parser.add_argument("--pyzlc_host", default="192.168.0.109")
    parser.add_argument("--pyzlc_group_name", default="DroidGroup")
    parser.add_argument("--pyzlc_group_port", type=int, default=7730)
    parser.add_argument("--policy_zmq_endpoint", default=None, help="Streaming ZMQ endpoint for the policy server.")
    parser.add_argument("--policy_zmq_timeout_ms", type=int, default=30000, help="ZMQ send/receive timeout.")
    parser.add_argument(
        "--continuous_min_execute_steps",
        type=int,
        default=0,
        help=(
            "Minimum number of actions to execute from each completed chunk before "
            "requesting the next replan; 0 preserves immediate continuous replanning."
        ),
    )
    parser.add_argument("--stop_after_first_release", action="store_true", help="Stop the episode after the first confirmed release.")
    parser.add_argument("--stop_after_release_steps", type=int, default=0, help="Extra policy steps to run after release before stopping.")
    parser.add_argument("--metrics_path", default=None, help="Append per-episode metrics as JSONL to this path.")
    parser.add_argument("--static_camera", default="static_cam", help="pyzlc static/base camera node name.")
    parser.add_argument("--wrist_camera", default="wrist_cam", help="pyzlc wrist camera node name.")
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
        args.robot_name,
        RemotePandaArm(args.robot_name),
        RemoteRobotiqGripper(args.robot_name),
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
        task=args.task,
        fps=args.fps,
        policy_transport="streaming_zmq",
        policy_zmq_endpoint=args.policy_zmq_endpoint,
        policy_zmq_timeout_ms=args.policy_zmq_timeout_ms,
        continuous_min_execute_steps=args.continuous_min_execute_steps,
        stop_after_first_release=args.stop_after_first_release,
        stop_after_release_steps=args.stop_after_release_steps,
        metrics_path=args.metrics_path,
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
