"""Optional Oopsie rollout recorder for Cartesian policy inference."""

from __future__ import annotations

import dataclasses
import datetime
import logging
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pyzlc


OUTCOME_TO_SUCCESS = {"success": 1.0, "fail": 0.0}
OUTCOME_TO_FILENAME_PREFIX = {
    "success": "success",
    "fail": "failure",
}
PENDING_LAB_ID = "pending_local_collection"


class OopsieRolloutRecorder:
    """Record one Oopsie step for every action applied by the policy loop."""

    def __init__(
        self,
        root_dir: str,
        robot_profile_path: str,
        operator_name: str,
        control_freq: int,
        task: str,
    ) -> None:
        try:
            from oopsie_data_tools.annotation_tool.episode_recorder import EpisodeRecorder
            from oopsie_data_tools.utils import contributor_config
            from oopsie_data_tools.utils.robot_profile.robot_profile import load_robot_profile
        except ImportError as exc:
            raise ImportError(
                "Oopsie recording was enabled, but oopsie_data_tools is not installed. "
                "Install it with `pip install oopsie-data-tools`."
            ) from exc

        self._episode_cls = EpisodeRecorder
        self.root_dir = Path(root_dir)
        self.operator_name = operator_name
        self.task = task
        profile = load_robot_profile(robot_profile_path)
        self.profile = dataclasses.replace(profile, control_freq=int(control_freq))
        self._contributor_config = contributor_config

        expected_cameras = {"static_cam", "wrist_cam"}
        missing = expected_cameras - set(self.profile.camera_names)
        if missing:
            raise ValueError(
                f"Oopsie profile is missing camera names {sorted(missing)}."
            )

        self.session_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self._writer_pool = ThreadPoolExecutor(max_workers=1)
        self._writer_futures: list[Future] = []
        self._episode = None
        self._closed = False
        self._used_names: set[str] = set()
        self._last_frames: Dict[str, np.ndarray] = {}

    @property
    def is_recording(self) -> bool:
        return self._episode is not None

    def start(self) -> None:
        if self._episode is not None:
            self.discard()
        self._episode = self._create_episode_recorder()
        base, attempt = self._episode.save_fname, 1
        while self._episode.save_fname in self._used_names:
            attempt += 1
            self._episode.save_fname = f"{base}_{attempt}"
        self._used_names.add(self._episode.save_fname)
        self._last_frames = {}
        pyzlc.info(
            f"Oopsie rollout started: "
            f"{self._episode.session_dir / self._episode.save_fname}"
        )

    def _create_episode_recorder(self):
        kwargs = {
            "robot_profile": self.profile,
            "data_root_dir": self.root_dir,
            "operator_name": self.operator_name,
            "resume_session_name": self.session_name,
        }
        try:
            return self._episode_cls(**kwargs)
        except RuntimeError as exc:
            if "Contributor config" not in str(exc) and "lab_id" not in str(exc):
                raise

        # EpisodeRecorder currently has no lab-id override for local-only collection.
        # Substitute a conspicuous placeholder only while constructing this recorder;
        # finalize_oopsie_lab_id.py replaces it after the supervisor runs `oopsie-data init`.
        original_reader = self._contributor_config.read_contributor_config
        self._contributor_config.read_contributor_config = lambda *args, **kwargs: (
            PENDING_LAB_ID,
            "",
        )
        try:
            episode = self._episode_cls(**kwargs)
        finally:
            self._contributor_config.read_contributor_config = original_reader
        logging.getLogger(__name__).warning(
            "Oopsie contributor config is not initialized. Recording locally with "
            "temporary lab_id=%r; finalize the files before validation or upload.",
            PENDING_LAB_ID,
        )
        return episode

    def save(self, outcome: Optional[str]) -> Optional[Path]:
        episode, self._episode = self._episode, None
        if episode is None:
            return None
        if episode.num_steps == 0:
            pyzlc.info("Oopsie rollout is empty; nothing was saved.")
            return None

        success = OUTCOME_TO_SUCCESS.get(outcome or "")
        label = outcome or "unlabeled"
        filename_prefix = OUTCOME_TO_FILENAME_PREFIX.get(label, "unlabeled")
        episode.save_fname = f"{filename_prefix}_{episode.save_fname}"
        h5_path = episode.session_dir / f"{episode.save_fname}.h5"
        future = self._writer_pool.submit(
            self._finish, episode, success, label, h5_path
        )
        self._writer_futures.append(future)
        return h5_path

    def _finish(self, episode, success, outcome: str, h5_path: Path) -> None:
        try:
            episode.finish_rollout(instruction=self.task, success=success)
            duration = episode.num_steps / self.profile.control_freq
            pyzlc.info(
                f"Oopsie rollout saved as {outcome.upper()}: {h5_path} "
                f"(steps={episode.num_steps}, {duration:.1f}s @ "
                f"{self.profile.control_freq} Hz)"
            )
        except Exception as exc:
            pyzlc.error(f"Saving Oopsie rollout {h5_path} failed: {exc}")

    def discard(self) -> None:
        if self._episode is None:
            return
        self._episode = None
        self._last_frames = {}
        pyzlc.info("Oopsie rollout discarded.")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.discard()
        self._writer_pool.shutdown(wait=True)

    def record_step(
        self,
        arm_state: dict,
        grip_state: dict,
        images: Dict[str, np.ndarray],
        action: np.ndarray,
    ) -> None:
        if self._episode is None:
            return

        frames: Dict[str, np.ndarray] = {}
        for camera_name in self.profile.camera_names:
            frame = images.get(camera_name)
            previous = self._last_frames.get(camera_name)
            if frame is None:
                frame = previous
            elif previous is not None and np.array_equal(frame, previous):
                frame = previous
            if frame is None:
                return
            frames[camera_name] = np.asarray(frame, dtype=np.uint8)
        self._last_frames = frames

        action = np.asarray(action, dtype=np.float64).reshape(-1)
        self._episode.record_step(
            observation={
                "robot_state": {
                    "joint_position": np.asarray(
                        arm_state["q"], dtype=np.float64
                    ).reshape(7),
                    "cartesian_position": np.asarray(
                        _extract_ee_pose(arm_state), dtype=np.float64
                    ).reshape(7),
                    "gripper_position": np.asarray(
                        [np.clip(float(grip_state.get("position", 0.0)), 0.0, 1.0)],
                        dtype=np.float64,
                    ),
                },
                "image_observation": frames,
            },
            action={
                "cartesian_position": action[:7],
                "gripper_binary": np.asarray([action[7]], dtype=np.float64),
            },
            additional_data={
                "joint_velocity": np.asarray(arm_state["dq"], dtype=np.float32),
                "external_joint_torque": np.asarray(
                    arm_state["tau_ext_hat_filtered"], dtype=np.float32
                ),
                "external_wrench": np.asarray(
                    arm_state["O_F_ext_hat_K"], dtype=np.float32
                ),
                "gripper_current": np.asarray(
                    [grip_state.get("current", np.nan)], dtype=np.float32
                ),
            },
        )


def _extract_ee_pose(arm_state: dict) -> np.ndarray:
    if "EE_pos" in arm_state and "EE_quat" in arm_state:
        return np.concatenate(
            [
                np.asarray(arm_state["EE_pos"], dtype=np.float64).reshape(3),
                np.asarray(arm_state["EE_quat"], dtype=np.float64).reshape(4),
            ]
        )
    transform = np.asarray(arm_state["O_T_EE"], dtype=np.float64).reshape(4, 4).T
    rotation = transform[:3, :3]
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        quat = np.asarray(
            [
                (rotation[2, 1] - rotation[1, 2]) / scale,
                (rotation[0, 2] - rotation[2, 0]) / scale,
                (rotation[1, 0] - rotation[0, 1]) / scale,
                0.25 * scale,
            ]
        )
    else:
        index = int(np.argmax(np.diag(rotation)))
        if index == 0:
            scale = np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
            quat = np.asarray([
                0.25 * scale,
                (rotation[0, 1] + rotation[1, 0]) / scale,
                (rotation[0, 2] + rotation[2, 0]) / scale,
                (rotation[2, 1] - rotation[1, 2]) / scale,
            ])
        elif index == 1:
            scale = np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
            quat = np.asarray([
                (rotation[0, 1] + rotation[1, 0]) / scale,
                0.25 * scale,
                (rotation[1, 2] + rotation[2, 1]) / scale,
                (rotation[0, 2] - rotation[2, 0]) / scale,
            ])
        else:
            scale = np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
            quat = np.asarray([
                (rotation[0, 2] + rotation[2, 0]) / scale,
                (rotation[1, 2] + rotation[2, 1]) / scale,
                0.25 * scale,
                (rotation[1, 0] - rotation[0, 1]) / scale,
            ])
    return np.concatenate([transform[:3, 3], quat])
