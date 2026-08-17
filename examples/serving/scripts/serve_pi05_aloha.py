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

"""Serve an RLinf OpenPI Pi0.5 RLT (Stage-1 SFT) checkpoint via the openpi-client WebSocket protocol.

This is the *deployment* entrypoint for models trained by ``examples/sft/train_vla_sft.py``
with ``model_type=openpi`` + ``use_rlt=True`` (e.g. the ALOHA "make a sandwich" Stage-1 SFT).

Contract guarantees
-------------------
1. The model is loaded exclusively through ``rlinf.models.embodiment.openpi.get_model``
   using the same resolved config as training (config_name=pi05_aloha_robotwin,
   action_horizon=16, action_chunk=16, action_env_dim=14, use_rlt=True, ...).
2. Actions are sampled with ``model.predict_action_batch(..., mode="eval")``, i.e. the
   exact same transform chain as training:
   AlohaInputs -> Normalize(norm_stats) -> Pi0.5/RLT -> Unnormalize -> AbsoluteActions -> AlohaOutputs.
   The server never re-applies delta/absolute or normalization conversions.
3. Norm stats are loaded strictly from ``<checkpoint_dir>/<repo_id>/norm_stats.json``.
   No copying, replacement or fallback to other norm stats is performed.
4. State dict load is checked for missing/unexpected keys and never silently ignored.
5. The server metadata reports checkpoint / config / norm-stats identities so the robot
   side can verify it talks to the intended model.

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
from openpi.serving import websocket_policy_server


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


def _ensure_batch(value: Any, *, expected_ndim_without_batch: int, name: str) -> np.ndarray:
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


def _normalise_observation(obs: dict[str, Any], default_prompt: str) -> dict[str, np.ndarray]:
    """Convert openpi-client observations to RLinf ``predict_action_batch`` input.

    Accepts either the ALOHA runtime keys (``images``/``state``/``prompt``) or the
    OpenPI keys (``observation/image``/``observation/wrist_image``/``observation/state``/``prompt``).
    Returns main_images [B,H,W,C] uint8, wrist_images [B,2,H,W,C] uint8, states [B,14],
    task_descriptions [B] str.
    """
    prompt = obs.get("prompt") or default_prompt

    if "images" in obs:
        images = obs["images"]
        if "cam_high" not in images:
            raise KeyError("ALOHA observation images must include 'cam_high'.")
        main_images = _ensure_batch(images["cam_high"], expected_ndim_without_batch=3, name="images.cam_high")
        left = _ensure_batch(images["cam_left_wrist"], expected_ndim_without_batch=3, name="images.cam_left_wrist")
        right = _ensure_batch(images["cam_right_wrist"], expected_ndim_without_batch=3, name="images.cam_right_wrist")
        wrist_images = np.stack([left[0], right[0]], axis=0)[None, ...]
        states = _ensure_batch(obs["state"], expected_ndim_without_batch=1, name="state")
    elif "observation/image" in obs:
        main_images = _ensure_batch(obs["observation/image"], expected_ndim_without_batch=3, name="observation/image")
        wrist_images = _ensure_batch(obs["observation/wrist_image"], expected_ndim_without_batch=4, name="observation/wrist_image")
        states = _ensure_batch(obs["observation/state"], expected_ndim_without_batch=1, name="observation/state")
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

    return {
        "main_images": main_images,
        "wrist_images": wrist_images,
        "states": states,
        "task_descriptions": [prompt] * states.shape[0],
        # ``obs_processor`` reads this key directly (None -> no extra view).
        "extra_view_images": None,
    }


def _build_model_cfg(args: argparse.Namespace) -> tuple[Any, Path, str, str, Any]:
    """Build the RLinf ``get_model`` config from the deployment YAML + CLI overrides."""
    conf = OmegaConf.load(args.config)
    checkpoint_dir = Path(args.checkpoint_dir or conf.checkpoint.dir).resolve()
    repo_id = args.repo_id or conf.checkpoint.repo_id
    default_prompt = args.default_prompt or conf.checkpoint.default_prompt

    model_openpi = dict(conf.model.openpi)
    model_openpi["config_name"] = conf.model.config_name

    model_cfg = OmegaConf.create(
        {
            "model_path": str(checkpoint_dir),
            "model_type": "openpi",
            "action_dim": conf.model.action_dim,
            "num_action_chunks": conf.model.num_action_chunks,
            "num_steps": conf.model.num_steps,
            "openpi_data": {"repo_id": repo_id},
            "openpi": model_openpi,
        }
    )
    return model_cfg, checkpoint_dir, repo_id, default_prompt, conf.server


def _check_state_dict(model: torch.nn.Module, weights_path: Path, *, strict: bool) -> None:
    """Compare checkpoint keys against the loaded model and never silently ignore diffs."""
    checkpoint = torch.load(weights_path, map_location="cpu")
    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(checkpoint.keys()) if isinstance(checkpoint, dict) else set()
    missing = sorted(ckpt_keys - model_keys)  # saved by training but absent in model
    unexpected = sorted(model_keys - ckpt_keys)  # present in model but absent in checkpoint
    if not missing and not unexpected:
        logging.info("State dict check OK: %d keys matched.", len(ckpt_keys))
        return
    message = (
        f"State dict mismatch: missing={len(missing)} unexpected={len(unexpected)}. "
        f"missing[:10]={missing[:10]} unexpected[:10]={unexpected[:10]}"
    )
    if strict:
        raise RuntimeError(message)
    logging.warning("NON-STRICT state dict check: %s", message)


class RLinfOpenPiPolicy:
    """Small policy wrapper compatible with the openpi-client WebSocket protocol."""

    def __init__(
        self,
        model_cfg: Any,
        *,
        checkpoint_dir: Path,
        repo_id: str,
        default_prompt: str,
        device: str,
        strict_load: bool,
    ):
        from rlinf.models.embodiment.openpi import get_model

        norm_stats_path = checkpoint_dir / repo_id / "norm_stats.json"
        if not norm_stats_path.exists():
            raise FileNotFoundError(
                f"Norm stats not found at {norm_stats_path}. The checkpoint dir must contain "
                f"<repo_id>/norm_stats.json; no copying/fallback is allowed."
            )
        weights_path = checkpoint_dir / "model_state_dict" / "full_weights.pt"
        if not weights_path.exists():
            raise FileNotFoundError(f"Missing model weights: {weights_path}")

        self._model = get_model(model_cfg)
        _check_state_dict(self._model, weights_path, strict=strict_load)
        self._model.to(device)
        self._model.eval()
        self._default_prompt = default_prompt

        self.metadata = {
            "model_type": "rlinf_openpi_pi05",
            "checkpoint_dir": str(checkpoint_dir),
            "repo_id": repo_id,
            "config_name": model_cfg.openpi.config_name,
            "action_horizon": int(model_cfg.num_action_chunks),
            "action_dim": int(model_cfg.action_dim),
            "num_steps": int(model_cfg.num_steps),
            "use_rlt": bool(model_cfg.openpi.use_rlt),
            "noise_method": model_cfg.openpi.noise_method,
            "default_prompt": default_prompt,
            "weights_sha256": _sha256_file(weights_path),
            "norm_stats_sha256": _sha256_file(norm_stats_path),
            "rlinf_commit": _git_head(Path(__file__).resolve().parents[3]),
        }

    @torch.no_grad()
    def infer(self, obs: dict[str, Any], **_: Any) -> dict[str, Any]:
        env_obs = _normalise_observation(obs, self._default_prompt)
        actions, _ = self._model.predict_action_batch(env_obs=env_obs, mode="eval")
        if actions.shape[0] == 1:
            actions = actions[0]
        return {"actions": actions}


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
    parser.add_argument("--default-prompt", default=None, help="Override task prompt.")
    parser.add_argument("--host", default=None, help="Bind host (default: from --config).")
    parser.add_argument("--port", type=int, default=None, help="Bind port (default: 8001).")
    parser.add_argument("--device", default=None, help="torch device (default: from --config).")
    parser.add_argument(
        "--strict-load",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Fail on state dict missing/unexpected keys (default: from --config).",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Load the model, run one random ALOHA observation, print action shape, and exit.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    args = _parse_args()

    model_cfg, checkpoint_dir, repo_id, default_prompt, server_conf = _build_model_cfg(args)
    host = args.host or server_conf.host
    port = args.port or server_conf.port
    device = args.device or server_conf.device
    strict_load = args.strict_load if args.strict_load is not None else server_conf.strict_load

    policy = RLinfOpenPiPolicy(
        model_cfg,
        checkpoint_dir=checkpoint_dir,
        repo_id=repo_id,
        default_prompt=default_prompt,
        device=device,
        strict_load=strict_load,
    )
    logging.info("Server metadata: %s", policy.metadata)

    if args.smoke_test:
        result = policy.infer(_random_aloha_observation(default_prompt))
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
        return

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Starting openpi-client policy server on %s:%s (%s)", host, port, local_ip)
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host=host,
        port=port,
        metadata=policy.metadata,
    )
    server.serve_forever()

if __name__ == "__main__":
    main()
