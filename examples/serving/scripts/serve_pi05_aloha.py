# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Serve a new-format RLinf Pi0.5 RLT checkpoint over WebSocket.

This entrypoint serves ALOHA Stage-1 checkpoints trained with
``model_type=openpi_rlinf`` and ``openpi.use_rlt=true``. A config must declare
its checkpoint format. Legacy checkpoints are rejected with instructions to
use the preserved legacy branch and environment.

Contract guarantees
-------------------
1. New checkpoints load through ``openpi_rlinf`` with strict state-dict checks
   and the same 16-step, 14-dimensional action contract used for training.
2. Actions use the same transform chain as training:
   AlohaInputs -> Normalize -> Pi0.5/RLT -> Unnormalize -> AbsoluteActions -> AlohaOutputs.
   The server never re-applies delta/absolute or normalization conversions.
3. Norm stats load strictly from ``<checkpoint_dir>/<repo_id>/norm_stats.json``.
   No copying, replacement or fallback to other norm stats is performed.
4. Every response is checked for shape ``[batch, 16, 14]`` and finite values.
5. Metadata reports the format, model type, checkpoint, norm stats, horizon,
   camera contract, and RLinf Git commit.

Protocol: openpi-client WebSocket (see openpi_client.websocket_client_policy).
Default port: 8001.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import socket
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf


def _sha256_file(path: Path) -> str:
    """SHA256 of a file, streamed in 1 MiB chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_head(repo_root: Path) -> str:
    """Best-effort short HEAD of the repository serving this model."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


def _as_numpy(value: Any) -> np.ndarray:
    """Convert tensor/array-like values to NumPy without changing layout."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _ensure_batch(
    value: Any, *, expected_ndim_without_batch: int, name: str
) -> np.ndarray:
    """Return ``value`` with a leading batch dimension."""
    array = _as_numpy(value)
    if array.ndim == expected_ndim_without_batch:
        return array[None, ...]
    if array.ndim == expected_ndim_without_batch + 1:
        return array
    raise ValueError(
        f"{name} must have {expected_ndim_without_batch} or "
        f"{expected_ndim_without_batch + 1} dims, got shape {array.shape}."
    )


def _normalise_observation(
    obs: dict[str, Any], default_prompt: str
) -> dict[str, np.ndarray]:
    """Convert openpi-client observations to RLinf ``predict_action_batch`` input.

    Accepts either the ALOHA runtime keys (``images``/``state``/``prompt``) or the
    OpenPI keys (``observation/image``/``observation/wrist_image``/``observation/state``/``prompt``).
    Returns main_images [B,H,W,C] uint8, wrist_images [B,2,H,W,C] uint8, states [B,14],
    task_descriptions [B] str.
    """
    prompt = obs.get("prompt") or default_prompt

    if "images" in obs:
        images = obs["images"]
        required_cameras = ("cam_high", "cam_left_wrist", "cam_right_wrist")
        missing_cameras = [key for key in required_cameras if key not in images]
        if missing_cameras:
            raise KeyError(
                "ALOHA observation is missing required camera inputs: "
                f"{missing_cameras}."
            )
        main_images = _ensure_batch(
            images["cam_high"], expected_ndim_without_batch=3, name="images.cam_high"
        )
        left = _ensure_batch(
            images["cam_left_wrist"],
            expected_ndim_without_batch=3,
            name="images.cam_left_wrist",
        )
        right = _ensure_batch(
            images["cam_right_wrist"],
            expected_ndim_without_batch=3,
            name="images.cam_right_wrist",
        )
        if left.shape[0] != right.shape[0]:
            raise ValueError(
                "ALOHA wrist-camera batch sizes do not match: "
                f"left={left.shape[0]}, right={right.shape[0]}."
            )
        wrist_images = np.stack([left, right], axis=1)
        states = _ensure_batch(
            obs["state"], expected_ndim_without_batch=1, name="state"
        )
    elif "observation/image" in obs:
        main_images = _ensure_batch(
            obs["observation/image"],
            expected_ndim_without_batch=3,
            name="observation/image",
        )
        wrist_images = _ensure_batch(
            obs["observation/wrist_image"],
            expected_ndim_without_batch=4,
            name="observation/wrist_image",
        )
        states = _ensure_batch(
            obs["observation/state"],
            expected_ndim_without_batch=1,
            name="observation/state",
        )
    else:
        raise KeyError(
            "Observation must use either ALOHA runtime keys ('images','state') or "
            "OpenPI keys ('observation/image','observation/wrist_image','observation/state')."
        )

    if not (main_images.shape[0] == wrist_images.shape[0] == states.shape[0]):
        raise ValueError(
            "Observation batch sizes do not match: "
            f"main={main_images.shape[0]}, wrist={wrist_images.shape[0]}, state={states.shape[0]}."
        )
    if wrist_images.shape[1] != 2:
        raise ValueError(
            f"ALOHA requires exactly two wrist cameras, got {wrist_images.shape}."
        )
    if states.shape[-1] != 14:
        raise ValueError(f"ALOHA state must have 14 values, got shape {states.shape}.")

    return {
        "main_images": main_images,
        "wrist_images": wrist_images,
        "states": states,
        "task_descriptions": [prompt] * states.shape[0],
        # ``obs_processor`` reads this key directly (None -> no extra view).
        "extra_view_images": None,
    }


def _build_model_cfg(args: argparse.Namespace) -> tuple[Any, Path, str, str, Any]:
    """Build a strict new-format model config or reject a legacy checkpoint."""
    conf = OmegaConf.load(args.config)
    checkpoint_format = str(args.checkpoint_format or conf.checkpoint.format).lower()
    if checkpoint_format not in {"legacy", "openpi_rlinf"}:
        raise ValueError(
            "checkpoint.format must be 'legacy' or 'openpi_rlinf', got "
            f"{checkpoint_format!r}."
        )
    if checkpoint_format == "legacy":
        raise RuntimeError(
            "Legacy Stage-1 checkpoints are intentionally isolated from this runtime. "
            "Serve this checkpoint from branch legacy/main-3eeb9265 at commit "
            "3eeb9265 with its preserved Python environment; do not load it through "
            "openpi_rlinf."
        )
    if not bool(conf.server.strict_load):
        raise ValueError(
            "openpi_rlinf policy serving requires server.strict_load=true."
        )

    checkpoint_dir = (
        Path(args.checkpoint_dir or conf.checkpoint.dir).expanduser().resolve()
    )
    repo_id = str(args.repo_id or conf.checkpoint.repo_id)
    default_prompt = str(args.default_prompt or conf.checkpoint.default_prompt)

    model_openpi = OmegaConf.to_container(conf.model.openpi, resolve=True)
    model_openpi["task"] = "eval"
    model_openpi["config_name"] = str(conf.model.config_name)

    action_horizon = int(conf.model.num_action_chunks)
    configured_horizons = {
        action_horizon,
        int(conf.model.openpi.action_horizon),
        int(conf.model.openpi.action_chunk),
    }
    if configured_horizons != {16}:
        raise ValueError(
            "ALOHA serving requires num_action_chunks=action_horizon="
            f"action_chunk=16, got {sorted(configured_horizons)}."
        )
    configured_action_dims = {
        int(conf.model.action_dim),
        int(conf.model.openpi.action_env_dim),
    }
    if configured_action_dims != {14}:
        raise ValueError(
            "ALOHA serving requires action_dim=action_env_dim=14, got "
            f"{sorted(configured_action_dims)}."
        )
    if int(conf.model.openpi.num_images_in_input) != 3:
        raise ValueError("ALOHA serving requires num_images_in_input=3.")

    model_cfg = OmegaConf.create(
        {
            "model_path": str(checkpoint_dir),
            "model_type": "openpi_rlinf",
            "precision": str(conf.model.precision),
            "is_lora": False,
            "pi05": True,
            "strict_load": True,
            "action_dim": int(conf.model.action_dim),
            "num_action_chunks": action_horizon,
            "num_steps": int(conf.model.num_steps),
            "openpi_data": {
                "repo_id": repo_id,
                "default_prompt": default_prompt,
            },
            "openpi": model_openpi,
        }
    )
    return model_cfg, checkpoint_dir, repo_id, default_prompt, conf.server


class RLinfOpenPiPolicy:
    """Strict openpi_rlinf policy for the openpi-client WebSocket protocol."""

    def __init__(
        self,
        model_cfg: Any,
        *,
        checkpoint_dir: Path,
        repo_id: str,
        default_prompt: str,
        device: str,
    ):
        from rlinf.models.embodiment.openpi_rlinf import get_model
        from rlinf.models.embodiment.openpi_rlinf.utils.rlt_utils import (
            resolve_full_weights,
        )

        norm_stats_path = checkpoint_dir / repo_id / "norm_stats.json"
        if not norm_stats_path.exists():
            raise FileNotFoundError(
                f"Norm stats not found at {norm_stats_path}. The checkpoint dir must contain "
                f"<repo_id>/norm_stats.json; no copying/fallback is allowed."
            )
        weights_path = resolve_full_weights(checkpoint_dir)
        if weights_path is None:
            raise FileNotFoundError(
                f"No RLinf full_weights.pt checkpoint exists under {checkpoint_dir}."
            )

        # model_cfg.strict_load=True makes the actual load_state_dict call strict.
        self._model = get_model(model_cfg)
        self._model.to(device)
        self._model.eval()
        self._default_prompt = default_prompt
        self._action_horizon = int(model_cfg.num_action_chunks)
        self._action_dim = int(model_cfg.action_dim)

        self.metadata = {
            "model_type": "openpi_rlinf",
            "checkpoint_format": "openpi_rlinf",
            "checkpoint_dir": str(checkpoint_dir),
            "repo_id": repo_id,
            "checkpoint_path": str(weights_path.resolve()),
            "config_name": model_cfg.openpi.config_name,
            "norm_stats_path": str(norm_stats_path.resolve()),
            "action_horizon": self._action_horizon,
            "action_dim": self._action_dim,
            "num_steps": int(model_cfg.num_steps),
            "use_rlt": bool(model_cfg.openpi.use_rlt),
            "noise_method": model_cfg.openpi.noise_method,
            "default_prompt": default_prompt,
            "weights_sha256": _sha256_file(weights_path),
            "norm_stats_sha256": _sha256_file(norm_stats_path),
            "image_keys": ["cam_high", "cam_left_wrist", "cam_right_wrist"],
            "rlinf_commit": _git_head(Path(__file__).resolve().parents[3]),
        }

    @torch.no_grad()
    def infer(self, obs: dict[str, Any], **_: Any) -> dict[str, Any]:
        env_obs = _normalise_observation(obs, self._default_prompt)
        actions, _ = self._model.predict_action_batch(env_obs=env_obs, mode="eval")
        actions = _as_numpy(actions)
        expected_shape = (
            env_obs["states"].shape[0],
            self._action_horizon,
            self._action_dim,
        )
        if actions.shape != expected_shape:
            raise RuntimeError(
                f"Policy returned actions with shape {actions.shape}; expected {expected_shape}."
            )
        if not np.isfinite(actions).all():
            raise RuntimeError("Policy returned non-finite ALOHA actions.")
        return {"actions": actions[0] if actions.shape[0] == 1 else actions}

    @torch.no_grad()
    def extract_stage2_features(
        self,
        obs: dict[str, Any],
    ) -> dict[str, np.ndarray]:
        """Extract and validate the frozen Stage-1 features used by Stage 2."""
        env_obs = _normalise_observation(obs, self._default_prompt)
        features = self._model.extract_rlt_obs(env_obs)
        batch_size = env_obs["states"].shape[0]
        expected_shapes = {
            "z_rl": (batch_size, 2048),
            "proprio": (batch_size, self._action_dim),
            "ref_chunk": (
                batch_size,
                self._action_horizon,
                self._action_dim,
            ),
        }
        validated: dict[str, np.ndarray] = {}
        for name, expected_shape in expected_shapes.items():
            if name not in features:
                raise RuntimeError(f"Stage-1 feature output is missing {name!r}.")
            value = _as_numpy(features[name])
            if value.shape != expected_shape:
                raise RuntimeError(
                    f"Stage-1 feature {name} has shape {value.shape}; "
                    f"expected {expected_shape}."
                )
            if not np.isfinite(value).all():
                raise RuntimeError(
                    f"Stage-1 feature {name} contains non-finite values."
                )
            validated[name] = value
        return validated


def _random_aloha_observation(prompt: str) -> dict[str, Any]:
    """Random observation used by the smoke test (no robot needed).

    Uses the ALOHA runtime format: HWC uint8 images (as sent by real robots).
    """
    return {
        "state": np.zeros((14,), dtype=np.float32),
        "images": {
            "cam_high": np.zeros((224, 224, 3), dtype=np.uint8),
            "cam_left_wrist": np.zeros((224, 224, 3), dtype=np.uint8),
            "cam_right_wrist": np.zeros((224, 224, 3), dtype=np.uint8),
        },
        "prompt": prompt,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="examples/serving/config/serve_pi05_aloha_sandwich.yaml",
        help="Deployment contract YAML (see examples/serving/config/).",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Override checkpoint dir (default: from --config). Must contain "
        "model_state_dict/full_weights.pt and <repo_id>/norm_stats.json.",
    )
    parser.add_argument("--repo-id", default=None, help="Override norm-stats asset id.")
    parser.add_argument(
        "--checkpoint-format",
        choices=("legacy", "openpi_rlinf"),
        default=None,
        help="Override checkpoint format. Legacy checkpoints are rejected here.",
    )
    parser.add_argument("--default-prompt", default=None, help="Override task prompt.")
    parser.add_argument(
        "--host", default=None, help="Bind host (default: from --config)."
    )
    parser.add_argument(
        "--port", type=int, default=None, help="Bind port (default: 8001)."
    )
    parser.add_argument(
        "--device", default=None, help="torch device (default: from --config)."
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Load the model, run one random ALOHA observation, print action shape, and exit.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s"
    )
    args = _parse_args()

    model_cfg, checkpoint_dir, repo_id, default_prompt, server_conf = _build_model_cfg(
        args
    )
    host = args.host or server_conf.host
    port = args.port or server_conf.port
    device = args.device or server_conf.device

    policy = RLinfOpenPiPolicy(
        model_cfg,
        checkpoint_dir=checkpoint_dir,
        repo_id=repo_id,
        default_prompt=default_prompt,
        device=device,
    )
    logging.info("Server metadata: %s", policy.metadata)

    if args.smoke_test:
        smoke_observation = _random_aloha_observation(default_prompt)
        result = policy.infer(smoke_observation)
        actions = np.asarray(result["actions"])
        logging.info("Smoke test action shape: %s", actions.shape)
        logging.info(
            "Smoke test action finite=%s min=%.6f max=%.6f mean=%.6f std=%.6f",
            bool(np.isfinite(actions).all()),
            float(np.nanmin(actions)),
            float(np.nanmax(actions)),
            float(np.nanmean(actions)),
            float(np.nanstd(actions)),
        )
        features = policy.extract_stage2_features(smoke_observation)
        logging.info(
            "Stage-2 feature shapes: %s",
            {name: value.shape for name, value in features.items()},
        )
        return

    hostname = socket.gethostname()
    from openpi.serving import websocket_policy_server

    local_ip = socket.gethostbyname(hostname)
    logging.info(
        "Starting openpi-client policy server on %s:%s (%s)", host, port, local_ip
    )
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host=host,
        port=port,
        metadata=policy.metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
