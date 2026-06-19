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
    parser.add_argument(
        "--policy_transport",
        choices=("pyzlc", "zmq", "streaming_zmq"),
        default="pyzlc",
        help="Transport to the remote policy server; streaming_zmq enables FASTER action streaming.",
    )
    parser.add_argument("--policy_zmq_endpoint", default=None, help="ZMQ endpoint for zmq/streaming_zmq policy transport.")
    parser.add_argument("--policy_zmq_timeout_ms", type=int, default=30000, help="ZMQ send/receive timeout.")
    parser.add_argument(
        "--max_position_step_m",
        type=float,
        default=0.0,
        help="Inference-side max Cartesian position step; disabled by default.",
    )
    parser.add_argument(
        "--max_rotation_step_rad",
        type=float,
        default=0.0,
        help="Inference-side max quaternion rotation step; disabled by default.",
    )
    parser.add_argument(
        "--execution_horizon",
        type=int,
        default=50,
        help="Actions executed from each chunk before requesting the next inference.",
    )
    parser.add_argument(
        "--gripper_open_confirm_steps",
        type=int,
        default=1,
        help="Consecutive open commands required to confirm release; 1 confirms immediately after close.",
    )
    parser.add_argument("--stop_after_first_release", action="store_true", help="Stop the episode after the first confirmed release.")
    parser.add_argument("--stop_after_release_steps", type=int, default=0, help="Extra policy steps to run after release before stopping.")
    parser.add_argument("--debug_image_dir", default=None, help="Directory for saved camera debug frames.")
    parser.add_argument("--debug_image_interval", type=int, default=25, help="Save one debug image pair every N policy steps.")
    parser.add_argument("--metrics_path", default=None, help="Append per-episode metrics as JSONL to this path.")
    parser.add_argument(
        "--reclose_after_release_min_motion_m",
        type=float,
        default=0.0,
        help="Suppress post-release close commands until this much motion from release pose",
    )
    parser.add_argument(
        "--faster_infer_time_schedule",
        choices=("const", "HAS"),
        default="const",
        help="Denoising schedule: const is fully denoised baseline; HAS enables FASTER horizon-aware streaming.",
    )
    parser.add_argument("--faster_alpha", type=float, default=1.0, help="HAS horizon curve exponent; larger values change how speedup varies over the horizon.")
    parser.add_argument("--faster_u0", type=float, default=0.9, help="HAS aggressiveness; lower values denoise early actions more and are usually smoother.")
    parser.add_argument(
        "--delay",
        type=int,
        default=0,
        help="Old-action prefix used while the next chunk becomes available.",
    )
    parser.add_argument(
        "--early_stop_actions",
        type=int,
        default=0,
        help="Stop HAS denoising after this many new actions are emitted; 0 disables early stopping.",
    )
    parser.add_argument(
        "--phase_fallback_schedule",
        choices=("none", "const", "HAS"),
        default="none",
        help="Optional schedule to use for a later task phase; const is safer for final placement.",
    )
    parser.add_argument(
        "--phase_fallback_trigger",
        choices=("after_gripper_close", "before_gripper_open"),
        default="after_gripper_close",
        help="When to switch to phase_fallback_schedule; before_gripper_open only switches near release.",
    )
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
        policy_transport=args.policy_transport,
        policy_zmq_endpoint=args.policy_zmq_endpoint,
        policy_zmq_timeout_ms=args.policy_zmq_timeout_ms,
        max_position_step_m=args.max_position_step_m,
        max_rotation_step_rad=args.max_rotation_step_rad,
        execution_horizon=args.execution_horizon,
        gripper_open_confirm_steps=args.gripper_open_confirm_steps,
        stop_after_first_release=args.stop_after_first_release,
        stop_after_release_steps=args.stop_after_release_steps,
        debug_image_dir=args.debug_image_dir,
        debug_image_interval=args.debug_image_interval,
        metrics_path=args.metrics_path,
        reclose_after_release_min_motion_m=args.reclose_after_release_min_motion_m,
        faster_infer_time_schedule=args.faster_infer_time_schedule,
        faster_alpha=args.faster_alpha,
        faster_u0=args.faster_u0,
        delay=args.delay,
        early_stop_actions=args.early_stop_actions,
        phase_fallback_schedule=args.phase_fallback_schedule,
        phase_fallback_trigger=args.phase_fallback_trigger,
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
