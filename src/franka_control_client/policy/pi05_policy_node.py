#!/usr/bin/env python

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pyzlc
import torch

from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.utils.random_utils import set_seed


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
    seed: int


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

        pyzlc.init(
            cfg.pyzlc_name,
            cfg.pyzlc_host,
            group_name=cfg.pyzlc_group_name,
            group_port=cfg.pyzlc_group_port,
        )
        pyzlc.register_subscriber_handler(cfg.obs_topic, self._on_observation)
        self._action_pub = pyzlc.Publisher(cfg.action_topic)

        self.train_cfg = self._load_train_cfg()
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
        if self.cfg.policy_dtype:
            cli_args.append(f"--policy.dtype={self.cfg.policy_dtype}")

        return TrainPipelineConfig.from_pretrained(
            pretrained_name_or_path=self.cfg.checkpoint_path,
            cli_args=cli_args,
        )

    def _load_policy_stack(self):
        policy = make_policy(
            cfg=self.train_cfg.policy,
            env_cfg=None,
            ds_meta=None,
            rename_map=getattr(self.train_cfg, "rename_map", None),
        )
        policy.eval()
        policy.to(self.cfg.device)

        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=self.train_cfg.policy,
            pretrained_path=self.train_cfg.policy.pretrained_path,
            preprocessor_overrides={"device_processor": {"device": self.cfg.device}},
        )
        pyzlc.info(f"Pi0.5 policy loaded on {self.cfg.device}")
        return policy, preprocessor, postprocessor

    def _on_observation(self, msg: Dict[str, Any]) -> None:
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
        if self._expected_state_dim is not None and state.shape[-1] != self._expected_state_dim:
            if state.shape[-1] > self._expected_state_dim:
                state = state[:, : self._expected_state_dim]
            else:
                state = np.pad(
                    state,
                    ((0, 0), (0, self._expected_state_dim - state.shape[-1])),
                    mode="constant",
                )

        observation: Dict[str, Any] = {
            "observation.state": torch.from_numpy(state),
        }

        for key, shape in self._expected_image_shapes.items():
            if key not in obs_msg:
                c, h, w = shape
                rgb = np.zeros((h, w, c), dtype=np.uint8)
            else:
                rgb = self._decode_image(obs_msg[key])
                rgb = self._resize_image(rgb, shape)
            observation[key] = (
                torch.from_numpy(np.ascontiguousarray(rgb))
                .float()
                .permute(2, 0, 1)
                .unsqueeze(0)
                / 255.0
            )

        task = obs_msg.get("task") or self.cfg.default_task
        if task:
            observation["task"] = task
        return observation

    def step(self) -> None:
        if self._latest_obs is None:
            return

        observation = self._build_observation(self._latest_obs)
        observation = self.preprocessor(observation)

        with torch.inference_mode():
            if hasattr(self.policy, "predict_action_chunk"):
                action_chunk = self.policy.predict_action_chunk(observation)
            else:
                action_chunk = self.policy.select_action(observation)

        if action_chunk.ndim == 2:
            action_chunk = action_chunk.unsqueeze(1)
        elif action_chunk.ndim == 1:
            action_chunk = action_chunk.reshape(1, 1, -1)
        if action_chunk.ndim != 3:
            raise RuntimeError(f"Unexpected action shape: {tuple(action_chunk.shape)}")

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
        self._action_pub.publish(
            {
                "timestamp": time.time(),
                "action": action_array.tolist(),
                "shape": list(action_array.shape),
            }
        )

    def run(self) -> None:
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
    parser.add_argument("--seed", type=int, default=1000)
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
