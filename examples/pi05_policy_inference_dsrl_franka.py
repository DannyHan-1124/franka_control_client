#!/usr/bin/env python

"""Run real Franka inference against ``pi05_policy_DSRL_node``."""

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
from franka_control_client.policy_inference.pi05_policy_inference_dsrl import (
    Pi05DSRLPolicyInference,
    Pi05DSRLPolicyInferenceConfig,
)
from franka_control_client.robotiq_gripper.robotiq_gripper import RemoteRobotiqGripper


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real Franka PI0.5 DSRL inference client")
    parser.add_argument("--task", required=True)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--control_hz", type=float, default=100.0)
    parser.add_argument("--pyzlc_name", default="pi05_dsrl_franka_client")
    parser.add_argument("--pyzlc_host", default="192.168.0.109")
    parser.add_argument("--pyzlc_group_name", default="DroidGroup")
    parser.add_argument("--pyzlc_group_port", type=int, default=7730)
    parser.add_argument("--policy_name", default="pi05_dsrl")
    parser.add_argument("--policy_zmq_endpoint", required=True)
    parser.add_argument("--policy_zmq_timeout_ms", type=int, default=30000)
    parser.add_argument("--chunk_replan_steps", type=int, default=50)
    parser.add_argument("--call_vla_after_actions", type=int, default=None)
    parser.add_argument("--inference_latency", type=int, default=0)
    parser.add_argument("--stop_after_first_release", action="store_true")
    parser.add_argument("--stop_after_release_steps", type=int, default=0)
    parser.add_argument("--close_gripper_on_reset", action="store_true")
    parser.add_argument("--metrics_path", default=None)
    parser.add_argument("--robot_name", default="FrankaPanda")
    parser.add_argument("--static_camera", default="static_cam")
    parser.add_argument("--wrist_camera", default="wrist_cam")
    args = parser.parse_args()
    if args.call_vla_after_actions is not None and args.call_vla_after_actions < 1:
        parser.error("--call_vla_after_actions must be >= 1")
    if args.inference_latency < 0:
        parser.error("--inference_latency must be >= 0")
    if args.call_vla_after_actions is None and args.inference_latency:
        parser.error("--inference_latency requires --call_vla_after_actions")
    if (
        args.call_vla_after_actions is not None
        and args.inference_latency > args.call_vla_after_actions
    ):
        parser.error("--inference_latency must not exceed --call_vla_after_actions")
    return args


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
        RemotePandaArm(args.robot_name),
        RemoteRobotiqGripper(args.robot_name),
    )
    control_pair = CartesianPolicyPandaRobotiqControlPair(
        follower.panda_arm, follower.robotiq_gripper, control_hz=args.control_hz
    )
    data_collectors: List[IRL_HardwareDataWrapper] = [
        ImageDataWrapper(CameraDevice(args.static_camera, preview=False), hw_name="static_cam"),
        ImageDataWrapper(CameraDevice(args.wrist_camera, preview=False), hw_name="wrist_cam"),
        PandaArmDataWrapper(follower.panda_arm),
        RobotiqGripperDataWrapper(follower.robotiq_gripper),
    ]
    cfg = Pi05DSRLPolicyInferenceConfig(
        policy_name=args.policy_name,
        task=args.task,
        fps=args.fps,
        policy_transport="zmq",
        policy_zmq_endpoint=args.policy_zmq_endpoint,
        policy_zmq_timeout_ms=args.policy_zmq_timeout_ms,
        chunk_replan_steps=args.chunk_replan_steps,
        call_vla_after_actions=args.call_vla_after_actions,
        inference_latency=args.inference_latency,
        stop_after_first_release=args.stop_after_first_release,
        stop_after_release_steps=args.stop_after_release_steps,
        close_gripper_on_reset=args.close_gripper_on_reset,
        metrics_path=args.metrics_path,
        run_metadata={"control_hz": args.control_hz, "robot_name": args.robot_name},
    )
    manager = Pi05DSRLPolicyInference(data_collectors, control_pair, cfg)
    try:
        manager.run()
    finally:
        pyzlc.shutdown()


if __name__ == "__main__":
    main()
