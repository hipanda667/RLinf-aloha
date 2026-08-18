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

"""Convert ALOHA HDF5 offline rollouts into an RLinf replay buffer.

This utility converts a directory of ``episode_*.hdf5`` files into the
``TrajectoryReplayBuffer`` checkpoint format consumed by RLinf off-policy/RLT
workers:

    metadata.json
    trajectory_index.json
    trajectory_<id>_<model_weights_id>.pt

The core conversion is:

1. Read images, proprio/state, actions, reward, and human-teleop from HDF5.
2. Run the frozen Stage 1 OpenPI RLT feature model via ``extract_rlt_obs``.
3. Build ``Trajectory`` objects with ``curr_obs``/``next_obs`` containing
   ``z_rl``, ``proprio``, and ``ref_chunk``.
4. Save the trajectories through ``TrajectoryReplayBuffer.save_checkpoint``.

The HDF5 key names below are intentionally centralized. Adjust them after
inspecting the real sandwich dataset schema.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

# Allow direct execution as:
#   python toolkits/replay_buffer/convert_hdf5_to_rlinf_buffer.py
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rlinf.data.schema import Trajectory  # noqa: E402
from rlinf.data.storage.replay import (  # noqa: E402
    TrajectoryReplayBuffer,
    validate_replay_checkpoint,
)
from rlinf.models import get_model  # noqa: E402

# ---------------------------------------------------------------------------
# User-editable configuration.
# ---------------------------------------------------------------------------

STAGE1_MODEL_PATH = os.environ.get("ALOHA_STAGE1_CHECKPOINT", "")
HDF5_DIR = os.environ.get("ALOHA_HDF5_DIR", "")
OUTPUT_BUFFER_DIR = os.environ.get("ALOHA_REPLAY_BUFFER_PATH", "")

DEVICE = os.environ.get(
    "CONVERSION_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"
)
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "8"))
MAX_EPISODES = int(os.environ.get("MAX_EPISODES", "0"))
EPISODE_IDS = tuple(
    int(value)
    for value in os.environ.get("EPISODE_IDS", "").split(",")
    if value.strip()
)

TASK_DESCRIPTION = os.environ.get("TASK_DESCRIPTION", "make a sandwich")
RLT_REPO_ID = os.environ.get("RLT_REPO_ID", "aloha_sandwich")

ACTION_DIM = int(os.environ.get("ACTION_DIM", "14"))
ACTION_CHUNK = int(os.environ.get("ACTION_CHUNK", "16"))
ACTION_STRIDE = int(os.environ.get("ACTION_STRIDE", "16"))

# How to convert scalar or single-element episode rewards into per-step rewards.
# "final" is usually correct for sparse episode-level success rewards;
# "broadcast" repeats the scalar at every frame.
SCALAR_REWARD_MODE = "final"

# Stage 1 OpenPI feature extractor config. Keep this aligned with
# aloha_sandwich_rlt_stage2_online_ac_mlp.yaml::rollout.rlt_feature_model.
RLT_FEATURE_MODEL_CFG = {
    "model_type": "openpi_rlinf",
    "precision": "bf16",
    "is_lora": False,
    "model_path": STAGE1_MODEL_PATH,
    "num_action_chunks": ACTION_CHUNK,
    "action_dim": ACTION_DIM,
    "num_steps": 4,
    "add_value_head": False,
    "openpi_data": {
        "repo_id": RLT_REPO_ID,
        "default_prompt": TASK_DESCRIPTION,
    },
    "openpi": {
        "task": "eval",
        "config_name": "pi05_aloha_robotwin_sandwich",
        "num_images_in_input": 3,
        "action_horizon": ACTION_CHUNK,
        "action_chunk": ACTION_CHUNK,
        "action_env_dim": ACTION_DIM,
        "num_steps": 4,
        "model_action_dim": 32,
        "paligemma_variant": "gemma_2b",
        "action_expert_variant": "gemma_300m",
        "max_token_len": 200,
        "discrete_state_input": True,
        "state_indices": [],
        "noise_method": "flow_noise",
        "noise_params": [0.16, 0.12, 200],
        "joint_logprob": True,
        "use_rlt": True,
        "rlt_prefix_seq_len": 1024,
        "rlt_image_only": False,
        "rlt_use_mask": True,
    },
}

# HDF5 key mapping. Update these if the real files use different paths.
STATE_KEY_CANDIDATES = (
    "observations/qpos",
    "observations/state",
    "observation/state",
    "state",
    "proprio",
)
ACTION_KEY_CANDIDATES = ("action", "actions")
REWARD_KEY_CANDIDATES = ("reward", "rewards")
HUMAN_TELEOP_KEY_CANDIDATES = (
    "human-teleop",
    "human_teleop",
    "intervene",
    "intervene_flag",
    "teleop_segments",
)
HUMAN_ACTION_KEY_CANDIDATES = (
    "human-teleop/action",
    "human_teleop/action",
    "intervene_action",
)

# These are packed into env_obs["main_images"], ["wrist_images"], and
# ["extra_view_images"], which OpenPI's obs_processor expects.
MAIN_IMAGE_KEY_CANDIDATES = (
    "observations/images/cam_high",
    "observation.images.cam_high",
    "images/cam_high",
    "cam_high",
)
WRIST_IMAGE_KEY_CANDIDATES = (
    "observations/images/cam_left_wrist",
    "observation.images.cam_left_wrist",
    "images/cam_left_wrist",
    "cam_left_wrist",
)
EXTRA_VIEW_IMAGE_KEY_CANDIDATES = (
    "observations/images/cam_right_wrist",
    "observation.images.cam_right_wrist",
    "images/cam_right_wrist",
    "cam_right_wrist",
)


@dataclass
class EpisodeArrays:
    """Raw per-frame arrays from one HDF5 episode."""

    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    human_flags: np.ndarray
    human_actions: np.ndarray | None
    main_images: np.ndarray
    wrist_images: np.ndarray | None
    extra_view_images: np.ndarray | None


def _require_h5py():
    try:
        import h5py  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "convert_hdf5_to_rlinf_buffer.py requires h5py. "
            "Install h5py in the environment used for conversion."
        ) from exc
    return h5py


def _read_first_existing(h5_file: Any, candidates: tuple[str, ...]) -> np.ndarray:
    for key in candidates:
        if key in h5_file:
            return np.asarray(h5_file[key])
    raise KeyError(
        f"None of the candidate keys exists in the HDF5 file: {', '.join(candidates)}"
    )


def _read_optional(h5_file: Any, candidates: tuple[str, ...]) -> np.ndarray | None:
    for key in candidates:
        if key in h5_file:
            return np.asarray(h5_file[key])
    return None


def _normalize_images(images: np.ndarray) -> np.ndarray:
    """Normalize image storage without eagerly decoding compressed frames.

    ALOHA HDF5 files may store each frame as JPEG bytes, giving shape [T]
    with dtype kind ``S``. Keep that compact representation here and decode
    selected minibatches in ``_image_tensor_for_indices``.
    """
    images = np.asarray(images)
    if images.ndim == 1 and images.dtype.kind in ("S", "O"):
        return images
    if images.ndim != 4:
        raise ValueError(
            "Expected image array [T,H,W,C], [T,C,H,W], or [T] JPEG bytes, "
            f"got shape={images.shape}, dtype={images.dtype}."
        )
    if images.shape[1] in (1, 3, 4) and images.shape[-1] not in (1, 3, 4):
        images = np.transpose(images, (0, 2, 3, 1))
    return np.ascontiguousarray(images)


def _decode_compressed_image(item: object) -> np.ndarray:
    from io import BytesIO

    from PIL import Image

    if isinstance(item, np.ndarray) and item.ndim == 0:
        item = item.item()
    encoded = bytes(item)
    try:
        with Image.open(BytesIO(encoded)) as image:
            return np.asarray(image.convert("RGB")).copy()
    except Exception as exc:
        raise ValueError("Failed to decode compressed image bytes from HDF5.") from exc


def _image_tensor_for_indices(
    images: np.ndarray | None,
    indices: np.ndarray,
) -> torch.Tensor | None:
    if images is None:
        return None
    selected = images[indices]
    if selected.ndim == 1 and selected.dtype.kind in ("S", "O"):
        decoded = [_decode_compressed_image(item) for item in selected]
        selected = np.stack(decoded, axis=0)
    elif selected.ndim == 4:
        selected = _normalize_images(selected)
    else:
        raise ValueError(
            "Unsupported selected image batch shape/dtype: "
            f"shape={selected.shape}, dtype={selected.dtype}."
        )
    return torch.as_tensor(np.ascontiguousarray(selected), device=DEVICE)


def _normalize_2d(name: str, value: np.ndarray, width: int | None = None) -> np.ndarray:
    value = np.asarray(value)
    if value.ndim == 1:
        value = value[:, None]
    if value.ndim > 2:
        value = value.reshape(value.shape[0], -1)
    if width is not None and value.shape[-1] < width:
        raise ValueError(
            f"{name} has width {value.shape[-1]}, expected at least {width}."
        )
    return value.astype(np.float32, copy=False)


def _normalize_rewards(rewards: np.ndarray, length: int) -> np.ndarray:
    rewards = np.asarray(rewards)

    # Some ALOHA datasets store reward as an episode-level scalar. Convert it
    # to the per-step reward tensor expected by RLinf trajectories.
    if rewards.ndim == 0 or rewards.size == 1:
        scalar_reward = float(rewards.reshape(-1)[0])
        if SCALAR_REWARD_MODE == "broadcast":
            return np.full((length,), scalar_reward, dtype=np.float32)
        if SCALAR_REWARD_MODE == "final":
            per_step = np.zeros((length,), dtype=np.float32)
            per_step[-1] = scalar_reward
            return per_step
        raise ValueError(
            "SCALAR_REWARD_MODE must be either 'final' or 'broadcast', "
            f"got {SCALAR_REWARD_MODE!r}."
        )

    if rewards.ndim > 1:
        rewards = rewards.reshape(rewards.shape[0], -1)[:, 0]
    if rewards.shape[0] not in {length, length + 1}:
        raise ValueError(
            f"reward length {rewards.shape[0]} must match the {length} transitions "
            "or include one final observation-aligned value."
        )
    return rewards[:length].astype(np.float32, copy=False)


def _parse_human_teleop(
    *,
    raw_human: np.ndarray | None,
    raw_human_actions: np.ndarray | None,
    default_actions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Normalize HDF5 human-teleop into per-step flags and optional actions.

    Supported layouts:
    - boolean/numeric flag: [T] or [T,1]
    - action payload: [T,ACTION_DIM], treated as human action where nonzero/finite
    - group-style separate action dataset via HUMAN_ACTION_KEY_CANDIDATES
    """
    num_steps = default_actions.shape[0]

    if raw_human is None:
        flags = np.zeros((num_steps,), dtype=bool)
        actions = raw_human_actions
    else:
        raw_human = np.asarray(raw_human)
        if raw_human.ndim == 2 and raw_human.shape[-1] == 2:
            # Some ALOHA files store human intervention as [start, end]
            # segments instead of a per-frame flag. Treat end as exclusive.
            flags = np.zeros((num_steps,), dtype=bool)
            for start, end in raw_human.astype(np.int64):
                start = max(0, int(start))
                end = min(num_steps, int(end))
                if end > start:
                    flags[start:end] = True
            actions = raw_human_actions
        elif raw_human.ndim == 1 or (raw_human.ndim == 2 and raw_human.shape[-1] == 1):
            flags = raw_human.reshape(num_steps).astype(bool)
            actions = raw_human_actions
        else:
            actions = raw_human.reshape(num_steps, -1).astype(np.float32, copy=False)
            if actions.shape[-1] < ACTION_DIM:
                raise ValueError(
                    "human-teleop action payload has width "
                    f"{actions.shape[-1]}, expected at least {ACTION_DIM}."
                )
            actions = actions[:, :ACTION_DIM]
            finite = np.isfinite(actions).all(axis=-1)
            nonzero = np.linalg.norm(actions, axis=-1) > 1e-8
            flags = finite & nonzero

    if actions is None:
        return flags.astype(bool, copy=False), None

    actions = _normalize_2d("human_actions", actions, width=ACTION_DIM)[:, :ACTION_DIM]
    if actions.shape[0] != num_steps:
        raise ValueError(
            f"human action length {actions.shape[0]} does not match episode length "
            f"{num_steps}."
        )
    return flags.astype(bool, copy=False), actions


def _trim_frame_aligned_human_data(
    value: np.ndarray | None,
    transition_length: int,
) -> np.ndarray | None:
    """Trim frame-aligned human data while preserving segment arrays."""
    if value is None:
        return None
    value = np.asarray(value)
    if value.ndim == 2 and value.shape[-1] == 2:
        return value
    return value[:transition_length]


def load_hdf5_episode(path: str) -> EpisodeArrays:
    """Load one ALOHA episode as observations plus aligned transitions.

    Both common ALOHA layouts are accepted: ``T`` actions with an unused final
    observation-aligned action, and ``T-1`` actions containing transitions
    only. In both cases the returned episode contains ``T`` observations and
    ``T-1`` transitions.
    """
    h5py = _require_h5py()
    with h5py.File(path, "r") as f:
        states = _normalize_2d("states", _read_first_existing(f, STATE_KEY_CANDIDATES))
        actions = _normalize_2d(
            "actions", _read_first_existing(f, ACTION_KEY_CANDIDATES), width=ACTION_DIM
        )
        raw_rewards = _read_first_existing(f, REWARD_KEY_CANDIDATES)
        raw_human = _read_optional(f, HUMAN_TELEOP_KEY_CANDIDATES)
        raw_human_actions = _read_optional(f, HUMAN_ACTION_KEY_CANDIDATES)

        main_images = _normalize_images(
            _read_first_existing(f, MAIN_IMAGE_KEY_CANDIDATES)
        )
        wrist_images = _read_optional(f, WRIST_IMAGE_KEY_CANDIDATES)
        extra_view_images = _read_optional(f, EXTRA_VIEW_IMAGE_KEY_CANDIDATES)
        if wrist_images is not None:
            wrist_images = _normalize_images(wrist_images)
        if extra_view_images is not None:
            extra_view_images = _normalize_images(extra_view_images)

    if states.shape[-1] != ACTION_DIM:
        raise ValueError(
            f"ALOHA states must have width {ACTION_DIM}, got {states.shape[-1]}."
        )
    if actions.shape[-1] != ACTION_DIM:
        raise ValueError(
            f"ALOHA actions must have width {ACTION_DIM}, got {actions.shape[-1]}."
        )
    if wrist_images is None or extra_view_images is None:
        raise KeyError(
            "ALOHA conversion requires cam_high, cam_left_wrist, and "
            "cam_right_wrist image streams."
        )

    observation_lengths = {
        "states": states.shape[0],
        "cam_high": main_images.shape[0],
        "cam_left_wrist": wrist_images.shape[0],
        "cam_right_wrist": extra_view_images.shape[0],
    }
    if len(set(observation_lengths.values())) != 1:
        raise ValueError(
            "ALOHA observation streams must have equal lengths, got "
            f"{observation_lengths}."
        )
    observation_length = next(iter(observation_lengths.values()))
    if observation_length < 2:
        raise ValueError(
            f"Episode {path} needs at least two observations, got {observation_length}."
        )

    # action[t] is the transition obs[t] -> obs[t+1]. The final frame has no
    # valid next observation. Some datasets nevertheless store an action at
    # that frame, while others already store exactly T-1 transition actions.
    transition_length = observation_length - 1
    if actions.shape[0] not in {transition_length, observation_length}:
        raise ValueError(
            "ALOHA action length must be T-1 or T for T observations, got "
            f"actions={actions.shape[0]}, observations={observation_length}."
        )
    actions = actions[:transition_length]
    rewards = _normalize_rewards(raw_rewards, length=transition_length)
    raw_human = _trim_frame_aligned_human_data(raw_human, transition_length)
    raw_human_actions = _trim_frame_aligned_human_data(
        raw_human_actions, transition_length
    )
    human_flags, human_actions = _parse_human_teleop(
        raw_human=raw_human,
        raw_human_actions=raw_human_actions,
        default_actions=actions,
    )

    return EpisodeArrays(
        states=states[:observation_length],
        actions=actions,
        rewards=rewards,
        human_flags=human_flags,
        human_actions=human_actions,
        main_images=main_images[:observation_length],
        wrist_images=(
            wrist_images[:observation_length] if wrist_images is not None else None
        ),
        extra_view_images=(
            extra_view_images[:observation_length]
            if extra_view_images is not None
            else None
        ),
    )


def _configured_path(value: str, *, env_var: str) -> Path:
    """Resolve a required path supplied through an environment variable."""
    if not value.strip():
        raise ValueError(
            f"{env_var} is required. Set it to an explicit local path; "
            "the converter intentionally has no machine-specific default."
        )
    return Path(value).expanduser()


def _stage1_full_weights_path() -> Path:
    checkpoint_dir = _configured_path(
        STAGE1_MODEL_PATH, env_var="ALOHA_STAGE1_CHECKPOINT"
    )
    candidates = (
        checkpoint_dir / "model_state_dict" / "full_weights.pt",
        checkpoint_dir / "actor" / "model_state_dict" / "full_weights.pt",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "Stage 1 full_weights.pt was not found under "
        f"{STAGE1_MODEL_PATH}. Checked: {candidates}"
    )


def _stage1_norm_stats_path() -> Path:
    path = (
        _configured_path(STAGE1_MODEL_PATH, env_var="ALOHA_STAGE1_CHECKPOINT")
        / RLT_REPO_ID
        / "norm_stats.json"
    )
    if not path.is_file():
        raise FileNotFoundError(
            f"Stage 1 norm stats do not exist at required path: {path}"
        )
    return path


def _sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_stage1_checkpoint_contract(model: torch.nn.Module) -> None:
    checkpoint = torch.load(
        _stage1_full_weights_path(),
        map_location="cpu",
        mmap=True,
        weights_only=True,
    )
    if not isinstance(checkpoint, dict):
        raise TypeError(
            "Stage 1 full_weights.pt must contain a state dict, got "
            f"{type(checkpoint).__name__}."
        )

    model_state = model.state_dict()
    checkpoint_keys = set(checkpoint)
    model_keys = set(model_state)
    missing = sorted(model_keys - checkpoint_keys)
    unexpected = sorted(checkpoint_keys - model_keys)
    mismatched = sorted(
        key
        for key in checkpoint_keys & model_keys
        if tuple(checkpoint[key].shape) != tuple(model_state[key].shape)
    )
    if missing or unexpected or mismatched:
        raise RuntimeError(
            "Stage 1 checkpoint contract mismatch: "
            f"missing={missing[:10]}, unexpected={unexpected[:10]}, "
            f"shape_mismatches={mismatched[:10]}."
        )


def build_rlt_feature_model() -> torch.nn.Module:
    if BATCH_SIZE <= 0:
        raise ValueError(f"BATCH_SIZE must be positive, got {BATCH_SIZE}.")
    if ACTION_CHUNK <= 0 or ACTION_STRIDE <= 0:
        raise ValueError(
            "ACTION_CHUNK and ACTION_STRIDE must be positive, got "
            f"{ACTION_CHUNK} and {ACTION_STRIDE}."
        )
    if DEVICE.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CONVERSION_DEVICE={DEVICE!r}, but CUDA is unavailable.")

    _stage1_norm_stats_path()
    cfg = OmegaConf.create(RLT_FEATURE_MODEL_CFG)
    model = get_model(cfg)
    _validate_stage1_checkpoint_contract(model)
    model.to(DEVICE)
    model.eval()
    model.requires_grad_(False)
    return model


def _slice_or_none(
    array: np.ndarray | None,
    indices: np.ndarray,
) -> torch.Tensor | None:
    if array is None:
        return None
    return torch.as_tensor(array[indices], device=DEVICE)


def make_env_obs(episode: EpisodeArrays, indices: np.ndarray) -> dict[str, Any]:
    batch_size = int(indices.shape[0])
    left_wrist = _image_tensor_for_indices(episode.wrist_images, indices)
    right_wrist = _image_tensor_for_indices(episode.extra_view_images, indices)
    if left_wrist is None or right_wrist is None:
        raise ValueError(
            "ALOHA Stage-1 feature extraction requires both wrist cameras."
        )
    # AlohaInputs expects one wrist tensor per sample with shape [2,H,W,C]:
    # index 0 is left wrist, index 1 is right wrist.
    wrist_images = torch.stack([left_wrist, right_wrist], dim=1)

    return {
        "main_images": _image_tensor_for_indices(episode.main_images, indices),
        "wrist_images": wrist_images,
        # Do not pass extra_view_images for ALOHA. The right wrist is folded
        # into wrist_images above because AlohaInputs consumes both wrists from
        # observation/wrist_image.
        "extra_view_images": None,
        "states": torch.as_tensor(episode.states[indices], device=DEVICE).float(),
        "task_descriptions": [TASK_DESCRIPTION] * batch_size,
    }


def extract_rlt_obs_for_indices(
    feature_model: torch.nn.Module,
    episode: EpisodeArrays,
    indices: np.ndarray,
) -> dict[str, torch.Tensor]:
    """Run Stage 1 feature extraction for selected frame indices."""
    chunks: dict[str, list[torch.Tensor]] = {"z_rl": [], "proprio": [], "ref_chunk": []}
    with torch.inference_mode():
        for start in range(0, len(indices), BATCH_SIZE):
            batch_indices = indices[start : start + BATCH_SIZE]
            rlt_obs = feature_model.extract_rlt_obs(
                make_env_obs(episode, batch_indices)
            )
            for key in chunks:
                chunks[key].append(rlt_obs[key].detach().cpu().float().contiguous())
    features = {key: torch.cat(values, dim=0) for key, values in chunks.items()}
    expected_shapes = {
        "z_rl": (2048,),
        "proprio": (ACTION_DIM,),
        "ref_chunk": (ACTION_CHUNK, ACTION_DIM),
    }
    for key, suffix in expected_shapes.items():
        value = features[key]
        if tuple(value.shape[-len(suffix) :]) != suffix:
            raise ValueError(
                f"Unexpected {key} shape {tuple(value.shape)}; "
                f"expected suffix {suffix}."
            )
        if not torch.isfinite(value).all():
            raise ValueError(f"Stage 1 produced non-finite values in {key}.")
    return features


def _pad_take(
    array: np.ndarray,
    start: int,
    length: int,
    *,
    pad_value: float | bool | None = None,
) -> np.ndarray:
    """Take a fixed window and pad its tail.

    ``pad_value=None`` repeats the final source row, which is appropriate for
    actions in a terminal chunk. Rewards and intervention flags must pass an
    explicit zero/False value so terminal labels are not duplicated.
    """
    end = start + length
    if end <= array.shape[0]:
        return array[start:end]
    if array.shape[0] == 0:
        raise ValueError("Cannot pad an empty array.")
    pad_count = end - array.shape[0]
    tail = array[start:]
    if pad_value is None:
        pad = np.repeat(array[-1:], pad_count, axis=0)
    else:
        pad = np.full(
            (pad_count, *array.shape[1:]),
            pad_value,
            dtype=array.dtype,
        )
    return np.concatenate([tail, pad], axis=0)


def build_chunked_trajectory(
    *,
    episode: EpisodeArrays,
    curr_features: dict[str, torch.Tensor],
    next_features: dict[str, torch.Tensor],
    starts: np.ndarray,
    model_weights_id: str,
) -> Trajectory:
    """Assemble one HDF5 episode into one RLinf Trajectory."""
    action_chunks = []
    reward_chunks = []
    termination_chunks = []
    truncation_chunks = []
    done_chunks = []
    intervene_chunks = []

    for start in starts.tolist():
        action_chunk = _pad_take(episode.actions, start, ACTION_CHUNK)
        reward_chunk = _pad_take(
            episode.rewards[:, None], start, ACTION_CHUNK, pad_value=0.0
        )[:, 0]
        human_flag_chunk = _pad_take(
            episode.human_flags[:, None].astype(np.bool_),
            start,
            ACTION_CHUNK,
            pad_value=False,
        )[:, 0]

        human_actions = (
            _pad_take(episode.human_actions, start, ACTION_CHUNK)
            if episode.human_actions is not None
            else action_chunk
        )
        final_action_chunk = np.where(
            human_flag_chunk[:, None],
            human_actions[:, :ACTION_DIM],
            action_chunk[:, :ACTION_DIM],
        )

        is_last_transition = start + ACTION_CHUNK >= episode.actions.shape[0]
        terminations = np.zeros((ACTION_CHUNK,), dtype=np.bool_)
        truncations = np.zeros((ACTION_CHUNK,), dtype=np.bool_)
        dones = np.zeros((ACTION_CHUNK,), dtype=np.bool_)
        if is_last_transition:
            valid_count = min(ACTION_CHUNK, episode.actions.shape[0] - start)
            terminal_index = valid_count - 1
            terminations[terminal_index] = True
            dones[terminal_index] = True

        action_chunks.append(final_action_chunk.reshape(-1))
        reward_chunks.append(reward_chunk)
        termination_chunks.append(terminations)
        truncation_chunks.append(truncations)
        done_chunks.append(dones)
        intervene_chunks.append(
            np.repeat(human_flag_chunk[:, None], ACTION_DIM, axis=1).reshape(-1)
        )

    curr_obs = {
        key: value.unsqueeze(1).cpu().contiguous()
        for key, value in curr_features.items()
    }
    next_obs = {
        key: value.unsqueeze(1).cpu().contiguous()
        for key, value in next_features.items()
    }

    trajectory = Trajectory(
        max_episode_length=int(len(starts)),
        model_weights_id=model_weights_id,
        actions=torch.as_tensor(np.asarray(action_chunks), dtype=torch.float32)
        .unsqueeze(1)
        .contiguous(),
        intervene_flags=torch.as_tensor(np.asarray(intervene_chunks), dtype=torch.bool)
        .unsqueeze(1)
        .contiguous(),
        rewards=torch.as_tensor(np.asarray(reward_chunks), dtype=torch.float32)
        .unsqueeze(1)
        .contiguous(),
        terminations=torch.as_tensor(np.asarray(termination_chunks), dtype=torch.bool)
        .unsqueeze(1)
        .contiguous(),
        truncations=torch.as_tensor(np.asarray(truncation_chunks), dtype=torch.bool)
        .unsqueeze(1)
        .contiguous(),
        dones=torch.as_tensor(np.asarray(done_chunks), dtype=torch.bool)
        .unsqueeze(1)
        .contiguous(),
        curr_obs=curr_obs,
        next_obs=next_obs,
    )
    return trajectory


def _transition_starts(num_transitions: int) -> np.ndarray:
    if num_transitions < 1:
        return np.empty((0,), dtype=np.int64)
    return np.arange(0, num_transitions, ACTION_STRIDE, dtype=np.int64)


def convert_episode(
    *,
    feature_model: torch.nn.Module,
    hdf5_path: str,
    model_weights_id: str,
) -> Trajectory | None:
    episode = load_hdf5_episode(hdf5_path)
    starts = _transition_starts(episode.actions.shape[0])
    if starts.size == 0:
        return None

    next_indices = np.minimum(starts + ACTION_CHUNK, episode.states.shape[0] - 1)

    curr_features = extract_rlt_obs_for_indices(feature_model, episode, starts)
    next_features = extract_rlt_obs_for_indices(feature_model, episode, next_indices)
    return build_chunked_trajectory(
        episode=episode,
        curr_features=curr_features,
        next_features=next_features,
        starts=starts,
        model_weights_id=model_weights_id,
    )


def _episode_number(path: str) -> int:
    match = re.search(r"episode_(\d+)\.hdf5$", path)
    if match is None:
        raise ValueError(f"Unexpected episode filename: {path}")
    return int(match.group(1))


def _select_hdf5_paths() -> tuple[list[str], int]:
    hdf5_dir = _configured_path(HDF5_DIR, env_var="ALOHA_HDF5_DIR")
    if not hdf5_dir.is_dir():
        raise FileNotFoundError(f"ALOHA HDF5 directory does not exist: {hdf5_dir}")
    all_paths = sorted(
        glob.glob(str(hdf5_dir / "episode_*.hdf5")),
        key=_episode_number,
    )
    if not all_paths:
        raise FileNotFoundError(f"No episode_*.hdf5 files found under {hdf5_dir}")

    selected = all_paths
    if EPISODE_IDS:
        by_id = {_episode_number(path): path for path in all_paths}
        missing = sorted(set(EPISODE_IDS) - set(by_id))
        if missing:
            raise FileNotFoundError(
                f"Requested EPISODE_IDS are missing under {hdf5_dir}: {missing}"
            )
        selected = [by_id[episode_id] for episode_id in EPISODE_IDS]
    if MAX_EPISODES > 0:
        selected = selected[:MAX_EPISODES]
    return selected, len(all_paths)


def _write_conversion_manifest(
    *,
    output_path: Path,
    hdf5_paths: list[str],
    full_weights_sha256: str,
    norm_stats_sha256: str,
    replay_stats: dict[str, float],
) -> None:
    manifest = {
        "version": 1,
        "stage1": {
            "model_path": str(
                _configured_path(
                    STAGE1_MODEL_PATH, env_var="ALOHA_STAGE1_CHECKPOINT"
                ).resolve()
            ),
            "full_weights_path": str(_stage1_full_weights_path().resolve()),
            "full_weights_sha256": full_weights_sha256,
            "norm_stats_path": str(_stage1_norm_stats_path().resolve()),
            "norm_stats_sha256": norm_stats_sha256,
            "repo_id": RLT_REPO_ID,
        },
        "source": {
            "hdf5_dir": str(
                _configured_path(HDF5_DIR, env_var="ALOHA_HDF5_DIR").resolve()
            ),
            "episodes": [
                {
                    "path": str(Path(path).resolve()),
                    "size_bytes": Path(path).stat().st_size,
                    "mtime_ns": Path(path).stat().st_mtime_ns,
                }
                for path in hdf5_paths
            ],
        },
        "conversion": {
            "device": DEVICE,
            "batch_size": BATCH_SIZE,
            "task_description": TASK_DESCRIPTION,
            "action_dim": ACTION_DIM,
            "action_chunk": ACTION_CHUNK,
            "action_stride": ACTION_STRIDE,
            "scalar_reward_mode": SCALAR_REWARD_MODE,
            "transition_semantics": "T_observations_to_T_minus_1_transitions",
        },
        "replay_buffer": replay_stats,
    }
    manifest_path = output_path / "conversion_manifest.json"
    temp_path = manifest_path.with_suffix(".json.tmp")
    temp_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    temp_path.replace(manifest_path)


def convert_directory() -> None:
    output_path = _configured_path(
        OUTPUT_BUFFER_DIR, env_var="ALOHA_REPLAY_BUFFER_PATH"
    )
    if output_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing replay-buffer path: {output_path}"
        )

    hdf5_paths, total_episodes = _select_hdf5_paths()
    print(f"Found {total_episodes} episodes; converting {len(hdf5_paths)} episodes.")
    print(
        "Conversion contract: "
        f"device={DEVICE}, batch_size={BATCH_SIZE}, action_dim={ACTION_DIM}, "
        f"action_chunk={ACTION_CHUNK}, action_stride={ACTION_STRIDE}, "
        f"repo_id={RLT_REPO_ID}."
    )

    full_weights_sha256 = _sha256_file(_stage1_full_weights_path())
    norm_stats_sha256 = _sha256_file(_stage1_norm_stats_path())
    model_weights_id = full_weights_sha256[:12]
    print(f"Stage 1 full weights SHA256: {full_weights_sha256}")
    print(f"Stage 1 norm stats SHA256: {norm_stats_sha256}")

    feature_model = build_rlt_feature_model()
    replay_buffer = TrajectoryReplayBuffer(
        seed=1234,
        enable_cache=True,
        cache_size=max(1, len(hdf5_paths)),
        sample_window_size=max(1, len(hdf5_paths)),
        auto_save=False,
        trajectory_format="pt",
    )

    for hdf5_path in tqdm(hdf5_paths, desc="Converting HDF5 episodes"):
        trajectory = convert_episode(
            feature_model=feature_model,
            hdf5_path=hdf5_path,
            model_weights_id=model_weights_id,
        )
        if trajectory is None:
            continue
        replay_buffer.add_trajectories([trajectory])

    replay_buffer.save_checkpoint(str(output_path))
    replay_stats = replay_buffer.get_stats()
    _write_conversion_manifest(
        output_path=output_path,
        hdf5_paths=hdf5_paths,
        full_weights_sha256=full_weights_sha256,
        norm_stats_sha256=norm_stats_sha256,
        replay_stats=replay_stats,
    )
    validate_replay_checkpoint(
        output_path,
        min_sample_count=1,
    )
    replay_buffer.close()
    print(f"Saved RLinf replay buffer to {output_path}")
    print(f"Stats: {replay_stats}")


if __name__ == "__main__":
    convert_directory()
