#!/usr/bin/env python

"""PI0.5 DSRL inference and online training node for a real Franka robot."""

from __future__ import annotations

import argparse
import logging
import os
import queue
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import zmq

WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
for dependency in (WORKSPACE_ROOT, WORKSPACE_ROOT / "DynamicVLA"):
    if str(dependency) not in sys.path:
        sys.path.insert(0, str(dependency))

from DOM_rl.adapters.pi05_dsrl_adapter import PI05DSRLAdapter  # noqa: E402
from DOM_rl.checkpoints import load_sac_checkpoint, save_sac_checkpoint  # noqa: E402
from DOM_rl.config import CompactEncoderConfig, DSRLConfig, RewardConfig  # noqa: E402
from DOM_rl.data_collect import Trajectory, TrajectoryRecorder  # noqa: E402
from DOM_rl.inference_client.chunk_trace import ChunkTraceRecorder  # noqa: E402
from DOM_rl.policies import SACPolicy  # noqa: E402
from DOM_rl.reward.basic_reward import BasicSparseReward  # noqa: E402
from DOM_rl.visualization import visualize_dsrl_chunk_noise  # noqa: E402


IMAGE_KEY_GROUPS: tuple[tuple[str, ...], ...] = (
    ("observation.images.static_cam", "observation.images.base_0_rgb"),
    ("observation.images.wrist_cam", "observation.images.left_wrist_0_rgb"),
)


@dataclass
class DSRLNodeConfig:
    weights: str
    bind: str
    output_dir: Path
    run_mode: str
    resume_checkpoint: Optional[Path]
    save_checkpoint_dir: Optional[Path]
    save_every_episodes: int
    checkpoint_mode: str
    rotation: str
    use_delta_action: bool
    streaming: bool
    chunk_trace_dir: Optional[Path]
    chunk_noise_viz_dir: Optional[Path]
    wandb_project: Optional[str]
    wandb_entity: Optional[str]
    wandb_run_name: Optional[str]
    wandb_group: Optional[str]
    wandb_tags: Optional[list[str]]
    wandb_mode: str
    wandb_dir: Path
    sac: DSRLConfig


class Pi05PolicyDSRLNode:
    """Owns the DSRL adapter; inference and training run on separate threads."""

    def __init__(self, cfg: DSRLNodeConfig) -> None:
        self.cfg = cfg
        self.adapter = PI05DSRLAdapter.from_pretrained(
            cfg.weights,
            rotation=cfg.rotation,
            use_delta_action=cfg.use_delta_action,
            streaming=cfg.streaming,
        )
        self.adapter.configure_sac(cfg.sac)
        self.adapter.set_deterministic_eval(cfg.run_mode == "inference")
        self.policy_updater = SACPolicy(cfg.sac, self.adapter)
        self.reward_fn = BasicSparseReward(
            RewardConfig(
                name="basic_sparse",
                step_penalty=0.0,
                success_reward=1.0,
                failure_reward=0.0,
            )
        )
        self.recorder = TrajectoryRecorder(cfg.output_dir)
        self.chunk_trace = (
            ChunkTraceRecorder(
                cfg.chunk_trace_dir,
                action_space="franka_quat8",
                action_names=(
                    "ee_pos_x", "ee_pos_y", "ee_pos_z",
                    "quat_x", "quat_y", "quat_z", "quat_w", "gripper",
                ),
                env_action_space="franka_quat8",
                env_action_names=(
                    "ee_pos_x", "ee_pos_y", "ee_pos_z",
                    "quat_x", "quat_y", "quat_z", "quat_w", "gripper",
                ),
            )
            if cfg.chunk_trace_dir is not None
            else None
        )
        self.wandb_run = self._init_wandb()
        self._model_lock = threading.RLock()
        self._train_queue: queue.Queue[tuple[int, Trajectory] | None] = queue.Queue()
        self._stop_event = threading.Event()
        self._trainer: threading.Thread | None = None
        self._trajectory: Trajectory | None = None
        self._episode_id: Any = None
        self._chunk_id = 0
        self._episodes_completed = 0

        if cfg.resume_checkpoint is not None:
            checkpoint = load_sac_checkpoint(
                cfg.resume_checkpoint,
                adapter=self.adapter,
                policy_updater=self.policy_updater,
                load_optimizer=cfg.run_mode == "online_train",
                expected_variant="dsrl",
            )
            self._episodes_completed = int(checkpoint.get("metadata", {}).get("episode", 0))
        if cfg.run_mode == "online_train":
            self._trainer = threading.Thread(
                target=self._trainer_loop, name="dsrl-trainer", daemon=True
            )
            self._trainer.start()

        self._context = zmq.Context.instance()
        self._socket = self._context.socket(zmq.REP)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.bind(cfg.bind)
        logging.info("DSRL Franka node bound to %s", cfg.bind)

    def _init_wandb(self):
        if not self.cfg.wandb_project:
            return None
        if self.cfg.run_mode != "online_train":
            logging.warning("W&B is configured outside online_train mode; episode metrics only will be logged.")
        directory = self.cfg.wandb_dir
        directory.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("WANDB_DIR", str(directory))
        os.environ.setdefault("WANDB_CACHE_DIR", str(directory / "cache"))
        try:
            import wandb
        except ImportError as exc:
            raise RuntimeError(
                "wandb is not installed; install it or omit --wandb_project"
            ) from exc
        run = wandb.init(
            project=self.cfg.wandb_project,
            entity=self.cfg.wandb_entity,
            name=self.cfg.wandb_run_name,
            group=self.cfg.wandb_group,
            tags=self.cfg.wandb_tags,
            mode=self.cfg.wandb_mode,
            dir=str(directory),
            config={
                "run_mode": self.cfg.run_mode,
                "weights": self.cfg.weights,
                "rotation": self.cfg.rotation,
                "sac": vars(self.cfg.sac),
            },
            settings=wandb.Settings(save_code=False, disable_git=True),
        )
        logging.info("Initialized W&B run: %s", getattr(run, "url", None))
        return run

    @staticmethod
    def _decode_image(value: Any) -> np.ndarray:
        if isinstance(value, np.ndarray):
            image = value
        elif isinstance(value, dict) and "rgb_data" in value:
            image = np.frombuffer(value["rgb_data"], dtype=np.uint8).reshape(
                int(value["height"]), int(value["width"]), int(value.get("channels", 3))
            )
        else:
            image = np.asarray(value, dtype=np.uint8)
        return np.ascontiguousarray(image)[None, ...]

    def _adapter_observation(self, message: dict[str, Any]) -> dict[str, Any]:
        state = np.asarray(message["observation.state"], dtype=np.float32).reshape(-1)
        if state.size < 8:
            raise ValueError(f"Expected Franka state [xyz,xyzw,gripper], got {state.size} values")
        observation: dict[str, Any] = {
            "observation.state": {
                "end_effector": {
                    "pos": state[None, :3],
                    "quat": state[None, 3:7],
                    "gripper": state[None, 7:8],
                }
            },
            "task": message.get("task", ""),
            "index": int(message.get("index", self._chunk_id)),
        }
        expected_input_keys_ordered = list(self.adapter.policy.config.input_features)
        expected_input_keys = set(expected_input_keys_ordered)
        expected_image_keys = [
            key for key in expected_input_keys_ordered if key.startswith("observation.images.")
        ]
        incoming_images = {
            key: value for key, value in message.items() if key.startswith("observation.images.")
        }
        mapped_images: dict[str, Any] = {}
        for key, value in incoming_images.items():
            if key in expected_input_keys:
                mapped_images[key] = value
                continue
            for group in IMAGE_KEY_GROUPS:
                if key in group:
                    target_key = next(
                        (candidate for candidate in group if candidate in expected_input_keys),
                        None,
                    )
                    if target_key is not None:
                        mapped_images[target_key] = value
                    break

        # Last-resort compatibility for checkpoints that use two custom camera
        # names: preserve the sender's camera order when the counts agree.
        if len(mapped_images) < len(expected_image_keys) and len(incoming_images) == len(
            expected_image_keys
        ):
            mapped_images = dict(zip(expected_image_keys, incoming_images.values()))

        for key, value in mapped_images.items():
            observation[key] = self._decode_image(value)
        return observation

    def _start_episode(self, message: dict[str, Any]) -> None:
        self._episode_id = message.get("episode_id", self._episodes_completed + 1)
        self._chunk_id = 0
        task = str(message.get("task", ""))
        with self._model_lock:
            self.adapter.start_episode(task)
        self._trajectory = Trajectory(
            task=task,
            episode_name=f"episode_{int(self._episode_id):06d}",
            metadata={"action_granularity": "chunk", "source": "real_franka"},
        )
        if self.chunk_trace is not None:
            self.chunk_trace.start_episode()

    def _infer(self, message: dict[str, Any]) -> dict[str, Any]:
        episode_id = message.get("episode_id", 0)
        if self._trajectory is None or episode_id != self._episode_id:
            self._start_episode(message)
        observation = self._adapter_observation(message)
        with self._model_lock, torch.no_grad():
            self.adapter.observe_environment_step(observation)
            output = self.adapter.sample_action_with_info(observation)
        if output.action is None:
            expected = list(self.adapter.policy.config.input_features)
            missing = [key for key in expected if key not in observation]
            raise RuntimeError(
                "DSRL adapter did not produce an action chunk; "
                f"missing policy inputs={missing}, expected={expected}, "
                f"received message keys={list(message)}, "
                f"mapped observation keys={list(observation)}"
            )
        action = np.asarray(output.action)
        if action.ndim == 3:
            action = action[0]
        if action.ndim != 2 or action.shape[-1] < 8:
            raise RuntimeError(f"Unexpected DSRL Franka action shape: {action.shape}")
        assert self._trajectory is not None
        self._trajectory.append(
            observation,
            output.recorded_action,
            prev_logprob=output.prev_logprob,
            prev_value=output.prev_value,
            forward_inputs=output.forward_inputs,
            action_granularity="chunk",
            valid_action_steps=0,
            chunk_id=self._chunk_id,
        )
        if self.chunk_trace is not None:
            self.chunk_trace.record_chunk(
                chunk_id=self._chunk_id,
                observation=observation,
                generated_action=action[:, :8],
                trace_action=output.recorded_action,
                sent_start_index=0,
            )
        response = {
            "timestamp": time.time(),
            "action": action[:, :8].tolist(),
            "shape": [int(action.shape[0]), 8],
            "chunk_id": self._chunk_id,
        }
        self._chunk_id += 1
        return response

    def _finish_episode(self, message: dict[str, Any]) -> dict[str, Any]:
        if self._trajectory is None:
            raise RuntimeError("Received episode_end without an active trajectory")
        success = bool(message["success"])
        executions = {
            index: int(count)
            for index, count in enumerate(message.get("chunk_executions", []))
        }
        self._trajectory.apply_chunk_executions(executions, granularity="chunk")
        if not self._trajectory.steps:
            raise RuntimeError("Episode contains no executed DSRL chunks")
        rewards = self.reward_fn.compute_episode_rewards(
            num_steps=len(self._trajectory.steps),
            success=success,
            step_counts=[step.valid_action_steps for step in self._trajectory.steps],
            trajectory=self._trajectory,
        )
        self._episodes_completed += 1
        self._trajectory.finalize(
            rewards=rewards,
            success=success,
            episode_name=f"episode_{self._episodes_completed:06d}",
            metadata={
                "keyboard_label": "success" if success else "failure",
                "chunk_executions": executions,
                "step_penalty": 0.0,
            },
        )
        path = self.recorder.save(self._trajectory)
        noise_viz_path = None
        if self.cfg.chunk_noise_viz_dir is not None:
            noise_viz_path = visualize_dsrl_chunk_noise(
                self._trajectory, self.cfg.chunk_noise_viz_dir
            )
        trace_path = None
        if self.chunk_trace is not None:
            trace_path = self.chunk_trace.finish_episode(
                episode_index=self._episodes_completed,
                observation={
                    "episode_id": self._episode_id,
                    "eps_name": self._trajectory.episode_name,
                    "success": success,
                    "chunk_executions": executions,
                    "chunk_start_indices": {chunk_id: 0 for chunk_id in executions},
                },
            )
        if self.wandb_run is not None:
            self.wandb_run.log(
                {
                    "episode/index": self._episodes_completed,
                    "episode/success": int(success),
                    "episode/chunks": len(self._trajectory.steps),
                    "episode/executed_actions": sum(executions.values()),
                },
            )
        if self.cfg.run_mode == "online_train":
            self._train_queue.put((self._episodes_completed, self._trajectory))
        logging.info(
            "Finalized episode %d: success=%s rewards=%s path=%s",
            self._episodes_completed, success, rewards, path,
        )
        self._trajectory = None
        return {
            "timestamp": time.time(),
            "action": [],
            "shape": [0, 8],
            "ack": True,
            "success": success,
            "path": str(path),
            "chunk_trace_path": None if trace_path is None else str(trace_path),
            "chunk_noise_viz_path": None if noise_viz_path is None else str(noise_viz_path),
        }

    def _discard_episode(self, message: dict[str, Any]) -> dict[str, Any]:
        episode_id = message.get("episode_id", self._episode_id)
        self._trajectory = None
        self._episode_id = None
        self._chunk_id = 0
        if self.chunk_trace is not None:
            self.chunk_trace.start_episode()
        with self._model_lock:
            self.adapter.reset()
        logging.info("Discarded DSRL episode %s without saving or training", episode_id)
        return {
            "timestamp": time.time(),
            "action": [],
            "shape": [0, 8],
            "ack": True,
            "discarded": True,
        }

    def _trainer_loop(self) -> None:
        while not self._stop_event.is_set():
            item = self._train_queue.get()
            if item is None:
                self._train_queue.task_done()
                break
            episode, trajectory = item
            try:
                with self._model_lock:
                    metrics = self.policy_updater.update([trajectory])
                    self._maybe_save_checkpoint(episode)
                if self.wandb_run is not None:
                    scalar_metrics = {
                        f"train/{key}": value
                        for key, value in metrics.items()
                        if isinstance(value, (int, float, np.number))
                    }
                    scalar_metrics["train/episode"] = episode
                    scalar_metrics["train/update_step"] = self.policy_updater.update_step
                    self.wandb_run.log(scalar_metrics)
                logging.info("DSRL trainer update: %s", metrics)
            except Exception:
                logging.exception("DSRL trainer failed for %s", trajectory.episode_name)
            finally:
                self._train_queue.task_done()

    def _maybe_save_checkpoint(self, episode: int) -> None:
        directory = self.cfg.save_checkpoint_dir
        every = self.cfg.save_every_episodes
        if directory is None or every <= 0 or episode % every:
            return
        path = directory / f"episode_{episode:06d}.pt"
        save_sac_checkpoint(
            path,
            adapter=self.adapter,
            sac_cfg=self.cfg.sac,
            policy_updater=self.policy_updater,
            mode=self.cfg.checkpoint_mode,
            metadata={
                "episode": episode,
                "run_mode": self.cfg.run_mode,
                "weights": self.cfg.weights,
                "reward": "basic_sparse_0_1",
            },
        )

    def run(self) -> None:
        try:
            while not self._stop_event.is_set():
                message = self._socket.recv_pyobj()
                try:
                    if message.get("dsrl_event") == "episode_end":
                        response = self._finish_episode(message)
                    elif message.get("dsrl_event") == "episode_discard":
                        response = self._discard_episode(message)
                    else:
                        response = self._infer(message)
                except Exception as exc:
                    logging.exception("DSRL request failed")
                    response = {"timestamp": time.time(), "error": str(exc)}
                self._socket.send_pyobj(response)
        finally:
            self.close()

    def close(self) -> None:
        if self._stop_event.is_set():
            return
        self._stop_event.set()
        if self._trainer is not None:
            self._train_queue.put(None)
            self._trainer.join(timeout=30)
        if self.wandb_run is not None:
            self.wandb_run.finish()
        self._socket.close(0)


def _parse_args() -> DSRLNodeConfig:
    parser = argparse.ArgumentParser(description="Real Franka PI0.5 DSRL node")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--bind", default="tcp://0.0.0.0:40023")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--run_mode", choices=("inference", "online_train"), default="online_train")
    parser.add_argument("--resume_checkpoint", type=Path, default=None)
    parser.add_argument("--save_checkpoint_dir", type=Path, default=None)
    parser.add_argument("--save_every_episodes", type=int, default=10)
    parser.add_argument("--checkpoint_mode", choices=("full", "trainable", "inference"), default="trainable")
    parser.add_argument("--rotation", choices=("auto", "quat", "rotvec", "euler"), default="quat")
    parser.add_argument("--delta", action="store_true")
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--chunk_trace_dir", type=Path, default=None)
    parser.add_argument("--dsrl_chunk_noise_viz_dir", type=Path, default=None)
    parser.add_argument("--wandb_project", default=None)
    parser.add_argument("--wandb_entity", default=None)
    parser.add_argument("--wandb_run_name", default=None)
    parser.add_argument("--wandb_group", default=None)
    parser.add_argument("--wandb_tags", nargs="*", default=None)
    parser.add_argument("--wandb_mode", choices=("online", "offline", "disabled"), default="online")
    parser.add_argument("--wandb_dir", type=Path, default=Path("/tmp/franka_dsrl_wandb"))
    parser.add_argument("--sac_gamma", type=float, default=0.99)
    parser.add_argument("--sac_tau", type=float, default=0.005)
    parser.add_argument("--sac_update_epochs", type=int, default=20)
    parser.add_argument("--sac_batch_size", type=int, default=64)
    parser.add_argument("--sac_min_buffer_size", type=int, default=100)
    parser.add_argument("--sac_replay_buffer_size", type=int, default=10000)
    parser.add_argument("--sac_actor_lr", type=float, default=1e-4)
    parser.add_argument("--sac_critic_lr", type=float, default=3e-4)
    parser.add_argument("--sac_alpha_lr", type=float, default=3e-4)
    parser.add_argument("--sac_initial_alpha", type=float, default=0.01)
    parser.add_argument("--sac_target_entropy", type=float, default=None)
    parser.add_argument("--sac_max_grad_norm", type=float, default=1.0)
    parser.add_argument("--dsrl_state_dim", type=int, default=8)
    parser.add_argument("--dsrl_action_noise_dim", type=int, default=32)
    parser.add_argument("--dsrl_action_magnitude", type=float, default=1.0)
    parser.add_argument("--dsrl_num_q_heads", type=int, default=10)
    parser.add_argument("--dsrl_agg_q", choices=("min", "mean"), default="mean")
    parser.add_argument("--dsrl_hidden_dims", type=int, nargs="+", default=[128, 128, 128])
    parser.add_argument("--dsrl_image_size", type=int, default=64)
    parser.add_argument("--dsrl_image_latent_dim", type=int, default=64)
    parser.add_argument("--dsrl_state_latent_dim", type=int, default=64)
    parser.add_argument("--dsrl_num_images", type=int, default=2)
    parser.add_argument("--dsrl_image_history_frames", type=int, default=1)
    parser.add_argument("--dsrl_image_history_stride", type=int, default=1)
    parser.add_argument("--dsrl_image_encoder_type", choices=("channel_concat", "temporal_conv"), default="channel_concat")
    args = parser.parse_args()
    if args.save_every_episodes < 0:
        parser.error("--save_every_episodes must be >= 0")
    sac = DSRLConfig(
        gamma=args.sac_gamma, tau=args.sac_tau, update_epochs=args.sac_update_epochs,
        batch_size=args.sac_batch_size, min_buffer_size=args.sac_min_buffer_size,
        replay_buffer_size=args.sac_replay_buffer_size, actor_lr=args.sac_actor_lr,
        critic_lr=args.sac_critic_lr, alpha_lr=args.sac_alpha_lr,
        initial_alpha=args.sac_initial_alpha, target_entropy=args.sac_target_entropy,
        max_grad_norm=args.sac_max_grad_norm, rollout_granularity="chunk",
        encoder=CompactEncoderConfig(
            state_dim=args.dsrl_state_dim, image_latent_dim=args.dsrl_image_latent_dim,
            state_latent_dim=args.dsrl_state_latent_dim, image_size=args.dsrl_image_size,
            num_images=args.dsrl_num_images, image_history_frames=args.dsrl_image_history_frames,
            image_history_stride=args.dsrl_image_history_stride,
            image_encoder_type=args.dsrl_image_encoder_type,
        ),
        action_noise_dim=args.dsrl_action_noise_dim,
        action_magnitude=args.dsrl_action_magnitude, num_q_heads=args.dsrl_num_q_heads,
        agg_q=args.dsrl_agg_q, hidden_dims=tuple(args.dsrl_hidden_dims), freeze_pi05=True,
    )
    return DSRLNodeConfig(
        weights=args.weights, bind=args.bind, output_dir=args.output_dir,
        run_mode=args.run_mode, resume_checkpoint=args.resume_checkpoint,
        save_checkpoint_dir=args.save_checkpoint_dir,
        save_every_episodes=args.save_every_episodes, checkpoint_mode=args.checkpoint_mode,
        rotation=args.rotation, use_delta_action=args.delta, streaming=args.streaming, sac=sac,
        chunk_trace_dir=args.chunk_trace_dir,
        chunk_noise_viz_dir=args.dsrl_chunk_noise_viz_dir,
        wandb_project=args.wandb_project, wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name, wandb_group=args.wandb_group,
        wandb_tags=args.wandb_tags, wandb_mode=args.wandb_mode,
        wandb_dir=args.wandb_dir,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(asctime)s %(message)s")
    node = Pi05PolicyDSRLNode(_parse_args())
    node.run()


if __name__ == "__main__":
    main()
