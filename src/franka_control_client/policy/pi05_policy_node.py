#!/usr/bin/env python

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pyzlc
import torch
import zmq

from lerobot.configs import RTCAttentionSchedule
from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.utils.random_utils import set_seed
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata


# Camera names used by the inference client/dataset may differ from the names
# stored in older policy checkpoints.  The keys are checkpoint feature names
# and the values are accepted aliases in incoming observation messages.
IMAGE_KEY_ALIASES: dict[str, tuple[str, ...]] = {
    "observation.images.static_cam": ("observation.images.base_0_rgb",),
    "observation.images.wrist_cam": ("observation.images.left_wrist_0_rgb",),
}


@dataclass
class Pi05NodeConfig:
    checkpoint_path: str
    dataset_path: Optional[str]
    device: str
    policy_dtype: Optional[str]
    obs_topic: str
    action_topic: str
    fps: float
    default_task: str
    pyzlc_name: str
    pyzlc_host: str
    pyzlc_group_name: str
    pyzlc_group_port: int
    direct_zmq_bind: Optional[str]
    seed: int
    rtc_enabled: bool
    rtc_execution_horizon: int
    rtc_max_guidance_weight: float
    rtc_prefix_attention_schedule: str
    rtc_debug: bool


class Pi05PolicyNode:
    """
    GPU-side Pi0.5 policy node.

    It subscribes to robot observations over pyzlc and publishes action chunks.
    The robot-side client applies each 8D Cartesian action sequentially.
    """

    def __init__(self, cfg: Pi05NodeConfig) -> None:
        self.cfg = cfg
        self._latest_obs: Optional[Dict[str, Any]] = None
        self._running = False
        self._policy_device = cfg.device

        self._action_pub = None
        self._direct_socket = None
        if cfg.direct_zmq_bind:
            self._ctx = zmq.Context.instance()
            self._direct_socket = self._ctx.socket(zmq.REP)
            self._direct_socket.setsockopt(zmq.LINGER, 0)
            self._direct_socket.bind(cfg.direct_zmq_bind)
            print(f"Direct ZMQ policy endpoint bound to {cfg.direct_zmq_bind}", flush=True)
        else:
            pyzlc.init(
                cfg.pyzlc_name,
                cfg.pyzlc_host,
                group_name=cfg.pyzlc_group_name,
                group_port=cfg.pyzlc_group_port,
            )
            pyzlc.register_subscriber_handler(cfg.obs_topic, self._on_observation)
            self._action_pub = pyzlc.Publisher(cfg.action_topic)

        self.train_cfg = self._load_train_cfg()
        self._validate_dataset_metadata()
        self.policy, self.preprocessor, self.postprocessor = self._load_policy_stack()
        self._expected_image_shapes = self._get_expected_image_shapes()
        self._expected_state_dim = self._get_expected_state_dim()

    def _load_train_cfg(self) -> TrainPipelineConfig:
        cli_args = [
            f"--policy.pretrained_path={self.cfg.checkpoint_path}",
            f"--policy.device={self.cfg.device}",
            "--dataset.image_transforms.enable=false",
        ]
        if self.cfg.dataset_path:
            cli_args.append(f"--dataset.root={self.cfg.dataset_path}")
            cli_args.append(f"--dataset.repo_id={self.cfg.dataset_path}")
        if self.cfg.policy_dtype:
            cli_args.append(f"--policy.dtype={self.cfg.policy_dtype}")

        train_cfg = TrainPipelineConfig.from_pretrained(
            pretrained_name_or_path=self.cfg.checkpoint_path,
            cli_args=cli_args,
        )
        self._configure_rtc(train_cfg)
        return train_cfg

    def _configure_rtc(self, train_cfg: TrainPipelineConfig) -> None:
        if not self.cfg.rtc_enabled:
            return

        try:
            schedule = RTCAttentionSchedule[self.cfg.rtc_prefix_attention_schedule.upper()]
        except KeyError as exc:
            valid = ", ".join(item.value.lower() for item in RTCAttentionSchedule)
            raise ValueError(
                f"Invalid RTC prefix attention schedule "
                f"{self.cfg.rtc_prefix_attention_schedule!r}; expected one of: {valid}"
            ) from exc

        train_cfg.policy.rtc_config = RTCConfig(
            enabled=True,
            prefix_attention_schedule=schedule,
            max_guidance_weight=float(self.cfg.rtc_max_guidance_weight),
            execution_horizon=int(self.cfg.rtc_execution_horizon),
            debug=bool(self.cfg.rtc_debug),
        )

    def _validate_dataset_metadata(self) -> None:
        if not self.cfg.dataset_path:
            return

        dataset = Path(self.cfg.dataset_path)
        episodes_dir = dataset / "meta" / "episodes"

        data_dir = dataset / "data"
        if not data_dir.exists():
            return

        import pandas as pd
        import pyarrow.parquet as pq

        episodes_path = episodes_dir / "chunk-000" / "file-000.parquet"
        episodes = pd.read_parquet(episodes_path)

        parts = []
        for parquet_path in sorted(data_dir.glob("*/*.parquet")):
            parts.append(pq.read_table(parquet_path, columns=["episode_index", "index"]).to_pandas())

        if not parts:
            return

        data = pd.concat(parts, ignore_index=True)
        actual = data.groupby("episode_index")["index"].agg(["min", "max", "count"]).reset_index()
        merged = episodes.merge(actual, on="episode_index", how="left")
        bad = merged[
            (merged["dataset_from_index"] != merged["min"])
            | (merged["dataset_to_index"] != merged["max"] + 1)
            | (merged["length"] != merged["count"])
        ]
        if len(bad) > 0:
            raise RuntimeError(
                f"Dataset episode metadata indexing is stale or inconsistent: {len(bad)} bad rows."
            )

    def _load_policy_stack(self):
        ds_meta = None
        if self.cfg.dataset_path:
            ds_meta = LeRobotDatasetMetadata(
                repo_id=self.cfg.dataset_path,
                root=self.cfg.dataset_path,
            )

        policy = make_policy(
            cfg=self.train_cfg.policy,
            env_cfg=None,
            ds_meta=ds_meta,
            rename_map=getattr(self.train_cfg, "rename_map", None),
        )
        if self.cfg.rtc_enabled and hasattr(policy, "init_rtc_processor"):
            policy.init_rtc_processor()
    
        policy.eval()

        device = self.cfg.device
        if device == "cuda" and not torch.cuda.is_available():
            pyzlc.info("CUDA unavailable; falling back to CPU")
            device = "cpu"
        self._policy_device = device

        policy.to(device)

        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=self.train_cfg.policy,
            pretrained_path=self.train_cfg.policy.pretrained_path,
            preprocessor_overrides={"device_processor": {"device": device}},
        )
        if self.cfg.direct_zmq_bind:
            print(f"Pi0.5 policy loaded on {self.cfg.device}", flush=True)
        else:
            pyzlc.info(f"Pi0.5 policy loaded on {self.cfg.device}")
        return policy, preprocessor, postprocessor

    def _on_observation(self, msg: Dict[str, Any]) -> None:
        print(f"Received observation keys: {list(msg.keys())}", flush=True)
        self._latest_obs = msg

    def _get_expected_image_shapes(self) -> dict[str, tuple[int, int, int]]:
        shapes: dict[str, tuple[int, int, int]] = {}
        input_feats = getattr(self.train_cfg.policy, "input_features", None)
        if isinstance(input_feats, dict):
            for key, feat in input_feats.items():
                if not str(key).startswith("observation.images."):
                    continue
                shape = tuple(feat.shape)
                if len(shape) == 3:
                    shapes[str(key)] = (int(shape[0]), int(shape[1]), int(shape[2]))
        return shapes

    def _get_expected_state_dim(self) -> Optional[int]:
        input_feats = getattr(self.train_cfg.policy, "input_features", None)
        if isinstance(input_feats, dict) and "observation.state" in input_feats:
            shape = tuple(input_feats["observation.state"].shape)
            if shape:
                return int(shape[-1])
        return None

    def _decode_image(self, image: Any) -> np.ndarray:
        if isinstance(image, np.ndarray):
            return np.ascontiguousarray(image)
        if isinstance(image, dict) and "rgb_data" in image:
            h = int(image["height"])
            w = int(image["width"])
            c = int(image.get("channels", 3))
            return np.frombuffer(image["rgb_data"], dtype=np.uint8).reshape(h, w, c).copy()
        if isinstance(image, list):
            return np.ascontiguousarray(np.asarray(image, dtype=np.uint8))
        raise ValueError(f"Unsupported image payload type: {type(image)!r}")

    def _resize_image(self, image: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
        c, h, w = shape
        if image.shape[:2] == (h, w):
            return image
        image_t = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
        resized = torch.nn.functional.interpolate(
            image_t, size=(h, w), mode="bilinear", align_corners=False
        )
        out = resized.squeeze(0).permute(1, 2, 0).byte().cpu().numpy()
        if out.shape[2] != c:
            raise ValueError(f"Image channels mismatch: expected {c}, got {out.shape[2]}")
        return out

    def _build_observation(self, obs_msg: Dict[str, Any]) -> Dict[str, Any]:
        state = np.asarray(obs_msg["observation.state"], dtype=np.float32).reshape(1, -1)
        
        observation: Dict[str, Any] = {
            "observation.state": torch.from_numpy(state),
        }

        for key, shape in self._expected_image_shapes.items():
            source_key = key
            if source_key not in obs_msg:
                source_key = next(
                    (alias for alias in IMAGE_KEY_ALIASES.get(key, ()) if alias in obs_msg),
                    "",
                )
            if not source_key:
                continue
            rgb = self._decode_image(obs_msg[source_key])
            rgb = self._resize_image(rgb, shape)
            image = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float() / 255.0
            observation[key] = image.unsqueeze(0)

        task = obs_msg.get("task") or self.cfg.default_task
        if task:
            observation["task"] = task
        return observation

    def _build_policy_kwargs(self, obs_msg: Dict[str, Any]) -> Dict[str, Any]:
        raw_kwargs = obs_msg.get("policy_kwargs") or {}
        if not isinstance(raw_kwargs, dict):
            raise ValueError("policy_kwargs must be a dictionary when provided.")

        policy_kwargs: Dict[str, Any] = {}
        for key, value in raw_kwargs.items():
            if key == "prev_chunk_left_over" and value is not None:
                prefix = np.asarray(value, dtype=np.float32)
                if prefix.ndim == 2:
                    prefix = prefix[None, :, :]
                if prefix.ndim != 3:
                    raise ValueError(
                        "prev_chunk_left_over must have shape (T, A) or (B, T, A)."
                    )
                policy_kwargs[key] = torch.from_numpy(prefix).to(self._policy_device)
            elif key in {"inference_delay", "execution_horizon"} and value is not None:
                policy_kwargs[key] = int(value)
            else:
                policy_kwargs[key] = value
        return policy_kwargs

    def _predict_action_msg(self, obs_msg: Dict[str, Any]) -> Dict[str, Any]:
        observation = self._build_observation(obs_msg)
        observation = self.preprocessor(observation)
        policy_kwargs = self._build_policy_kwargs(obs_msg)
        rtc_request = self.cfg.rtc_enabled and policy_kwargs.get("prev_chunk_left_over") is not None

        context = torch.enable_grad() if rtc_request else torch.inference_mode()
        with context:
            if hasattr(self.policy, "predict_action_chunk"):
                action_chunk = self.policy.predict_action_chunk(observation, **policy_kwargs)
            else:
                action_chunk = self.policy.select_action(observation)

        if action_chunk.ndim == 2:
            action_chunk = action_chunk.unsqueeze(0)
        elif action_chunk.ndim == 1:
            action_chunk = action_chunk.reshape(1, 1, -1)
        if action_chunk.ndim != 3:
            raise RuntimeError(f"Unexpected action shape: {tuple(action_chunk.shape)}")

        raw_actions = action_chunk.detach().float().cpu().numpy()
        processed_actions = []
        for idx in range(action_chunk.shape[1]):
            single = action_chunk[:, idx, :]
            try:
                processed = self.postprocessor(single)
            except Exception:
                processed = self.postprocessor(single[:, :8])
            if processed.ndim == 2:
                processed = processed[0]
            processed_actions.append(processed.detach().float().cpu().numpy()[:8])

        action_array = np.stack(processed_actions, axis=0)
        return {
            "timestamp": time.time(),
            "action": action_array.tolist(),
            "shape": list(action_array.shape),
            "action_raw": raw_actions.tolist(),
            "raw_shape": list(raw_actions.shape),
        }

    def step(self) -> None:
        if self._latest_obs is None:
            return
        if self._action_pub is None:
            raise RuntimeError("pyzlc action publisher is unavailable in direct ZMQ mode.")

        self._action_pub.publish(self._predict_action_msg(self._latest_obs))

    def run_direct_zmq(self) -> None:
        if self._direct_socket is None:
            raise RuntimeError("Direct ZMQ socket is not configured.")

        self._running = True
        while self._running:
            obs_msg = self._direct_socket.recv_pyobj()
            try:
                self._direct_socket.send_pyobj(self._predict_action_msg(obs_msg))
            except Exception as exc:
                print(f"Pi0.5 direct policy step failed: {exc}", flush=True)
                self._direct_socket.send_pyobj(
                    {
                        "timestamp": time.time(),
                        "action": [[0.0, 0.0, 0.0, -2.15, 0.0, 2.15, 0.0, 0.0]],
                        "shape": [1, 8],
                        "error": str(exc),
                    }
                )

    def run(self) -> None:
        if self.cfg.direct_zmq_bind:
            self.run_direct_zmq()
            return

        self._running = True
        dt = 1.0 / self.cfg.fps if self.cfg.fps > 0 else 0.0
        while self._running:
            start = time.perf_counter()
            try:
                self.step()
            except Exception as exc:
                pyzlc.error(f"Pi0.5 policy step failed: {exc}")
            if dt > 0:
                elapsed = time.perf_counter() - start
                if elapsed < dt:
                    pyzlc.sleep(dt - elapsed)


def _parse_args() -> Pi05NodeConfig:
    parser = argparse.ArgumentParser(description="Pi0.5 policy node over pyzlc.")
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--dataset_path", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--policy_dtype", default="bfloat16")
    parser.add_argument("--obs_topic", default="pi05/observation")
    parser.add_argument("--action_topic", default="pi05/action")
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--default_task", default="")
    parser.add_argument("--pyzlc_name", default="pi05_policy_node")
    parser.add_argument("--pyzlc_host", default="0.0.0.0")
    parser.add_argument("--pyzlc_group_name", default="DroidGroup")
    parser.add_argument("--pyzlc_group_port", type=int, default=7730)
    parser.add_argument(
        "--direct_zmq_bind",
        default=None,
        help="Optional direct REQ/REP policy endpoint, e.g. tcp://127.0.0.1:40023.",
    )
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument(
        "--rtc_enabled",
        action="store_true",
        help="Enable Real-Time Chunking guidance for requests that include RTC policy kwargs.",
    )
    parser.add_argument(
        "--rtc_execution_horizon",
        type=int,
        default=25,
        help="RTC execution horizon s: chunk index where the client switches/plans the next overlap.",
    )
    parser.add_argument(
        "--rtc_max_guidance_weight",
        type=float,
        default=5.0,
        help="RTC guidance weight clip; paper uses 5 for real-world experiments.",
    )
    parser.add_argument(
        "--rtc_prefix_attention_schedule",
        default="exp",
        choices=("zeros", "ones", "linear", "exp"),
        help="RTC prefix weight schedule; paper uses exponential prefix weights.",
    )
    parser.add_argument("--rtc_debug", action="store_true")
    args = parser.parse_args()
    return Pi05NodeConfig(**vars(args))


def main() -> None:
    cfg = _parse_args()
    set_seed(cfg.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    node = Pi05PolicyNode(cfg)
    try:
        node.run()
    finally:
        pyzlc.shutdown()


if __name__ == "__main__":
    main()
