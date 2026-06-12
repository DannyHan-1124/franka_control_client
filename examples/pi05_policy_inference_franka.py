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
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--control_hz", type=float, default=100.0)
    parser.add_argument("--pyzlc_name", default="pi05_franka_client")
    parser.add_argument("--pyzlc_host", default="192.168.0.109")
    parser.add_argument("--pyzlc_group_name", default="DroidGroup")
    parser.add_argument("--pyzlc_group_port", type=int, default=7730)
    parser.add_argument("--policy_transport", choices=("pyzlc", "zmq", "streaming_zmq"), default="pyzlc")
    parser.add_argument("--policy_zmq_endpoint", default=None)
    parser.add_argument("--policy_zmq_timeout_ms", type=int, default=30000)
    parser.add_argument("--max_position_step_m", type=float, default=0.005)
    parser.add_argument("--max_rotation_step_rad", type=float, default=0.05)
    parser.add_argument("--chunk_replan_steps", type=int, default=50)
    parser.add_argument("--gripper_open_confirm_steps", type=int, default=12)
    parser.add_argument("--stop_after_first_release", action="store_true")
    parser.add_argument("--stop_after_release_steps", type=int, default=0)
    parser.add_argument("--debug_image_dir", default=None)
    parser.add_argument("--debug_image_interval", type=int, default=25)
    parser.add_argument("--reclose_after_release_min_motion_m", type=float, default=0.08)
    parser.add_argument("--faster_infer_time_schedule", choices=("const", "HAS"), default="const")
    parser.add_argument("--faster_alpha", type=float, default=1.0)
    parser.add_argument("--faster_u0", type=float, default=0.9)
    parser.add_argument("--faster_delay_steps", type=int, default=0)
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
        task=args.task,
        fps=args.fps,
        policy_transport=args.policy_transport,
        policy_zmq_endpoint=args.policy_zmq_endpoint,
        policy_zmq_timeout_ms=args.policy_zmq_timeout_ms,
        max_position_step_m=args.max_position_step_m,
        max_rotation_step_rad=args.max_rotation_step_rad,
        chunk_replan_steps=args.chunk_replan_steps,
        gripper_open_confirm_steps=args.gripper_open_confirm_steps,
        stop_after_first_release=args.stop_after_first_release,
        stop_after_release_steps=args.stop_after_release_steps,
        debug_image_dir=args.debug_image_dir,
        debug_image_interval=args.debug_image_interval,
        reclose_after_release_min_motion_m=args.reclose_after_release_min_motion_m,
        faster_infer_time_schedule=args.faster_infer_time_schedule,
        faster_alpha=args.faster_alpha,
        faster_u0=args.faster_u0,
        faster_delay_steps=args.faster_delay_steps,
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
