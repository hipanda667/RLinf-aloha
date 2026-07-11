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
import os
import sys
import uuid
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

from rlinf.data.embodied_io_struct import Trajectory
from rlinf.data.replay_buffer import TrajectoryReplayBuffer
from rlinf.models import get_model


# ---------------------------------------------------------------------------
# User-editable configuration.
# ---------------------------------------------------------------------------

# Replace this with the final Stage 1 checkpoint directory when it finishes.
STAGE1_MODEL_PATH = (
    "/inspire/qb-ilm/project/robot-reasoning/czxs253130583/yushun/RLinf-worktree-rltoken-anhao/results/rlt_stage1_sft_pi05_sandwich_h200_2gpu/rlt_stage1_sft_pi05_sandwich_h100_2gpu/checkpoints/global_step_20000/actor"
)

HDF5_DIR = (
    "/inspire/qb-ilm/project/robot-reasoning/czxs253130583/yushun/"
    "aloha-data/sandwich_rl"
)

OUTPUT_BUFFER_DIR = os.environ.get(
    "OUTPUT_BUFFER_DIR",
    (
        "/inspire/qb-ilm/project/robot-reasoning/czxs253130583/yushun/"
        "RLinf-worktree-rltoken-anhao/results/"
        "rlt_stage2_sandwich_replay_buffer"
    ),
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8

# new add
MAX_EPISODES = int(os.environ.get("MAX_EPISODES", "0"))

TASK_DESCRIPTION = "make a sandwich"

# RLT Stage2 config uses 10 action chunks and 14D ALOHA actions.
ACTION_DIM = 14
ACTION_CHUNK = 10
ACTION_STRIDE = 10

# How to convert scalar or single-element episode rewards into per-step rewards.
# "final" is usually correct for sparse episode-level success rewards;
# "broadcast" repeats the scalar at every frame.
SCALAR_REWARD_MODE = "final"

# Stage 1 OpenPI feature extractor config. Keep this aligned with
# examples/embodiment/config/rlt_stage2_ac_mlp.yaml::rollout.rlt_feature_model.
RLT_FEATURE_MODEL_CFG = {
    "model_type": "openpi",
    "precision": None,
    "is_lora": False,
    "model_path": STAGE1_MODEL_PATH,
    "openpi_data": {
        "repo_id": "pi05_sandwich_new_all",
    },
    "openpi": {
        "config_name": "pi05_aloha_robotwin",
        "num_images_in_input": 3,
        "action_chunk": 50,
        "num_steps": 4,
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
        "None of the candidate keys exists in the HDF5 file: "
        f"{', '.join(candidates)}"
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
    import cv2

    if isinstance(item, np.ndarray) and item.ndim == 0:
        item = item.item()
    encoded = bytes(item)
    array = np.frombuffer(encoded, dtype=np.uint8)
    image_bgr = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError("Failed to decode compressed image bytes from HDF5.")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return image_rgb


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
    if rewards.shape[0] != length:
        raise ValueError(
            f"reward length {rewards.shape[0]} does not match episode length {length}."
        )
    return rewards.astype(np.float32, copy=False)


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
        elif raw_human.ndim == 1 or (
            raw_human.ndim == 2 and raw_human.shape[-1] == 1
        ):
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


def load_hdf5_episode(path: str) -> EpisodeArrays:
    """Load one ALOHA HDF5 episode into normalized arrays."""
    h5py = _require_h5py()
    with h5py.File(path, "r") as f:
        states = _normalize_2d("states", _read_first_existing(f, STATE_KEY_CANDIDATES))
        actions = _normalize_2d(
            "actions", _read_first_existing(f, ACTION_KEY_CANDIDATES), width=ACTION_DIM
        )[:, :ACTION_DIM]
        rewards = _normalize_rewards(
            _read_first_existing(f, REWARD_KEY_CANDIDATES),
            length=actions.shape[0],
        )
        raw_human = _read_optional(f, HUMAN_TELEOP_KEY_CANDIDATES)
        raw_human_actions = _read_optional(f, HUMAN_ACTION_KEY_CANDIDATES)
        human_flags, human_actions = _parse_human_teleop(
            raw_human=raw_human,
            raw_human_actions=raw_human_actions,
            default_actions=actions,
        )

        main_images = _normalize_images(
            _read_first_existing(f, MAIN_IMAGE_KEY_CANDIDATES)
        )
        wrist_images = _read_optional(f, WRIST_IMAGE_KEY_CANDIDATES)
        extra_view_images = _read_optional(f, EXTRA_VIEW_IMAGE_KEY_CANDIDATES)
        if wrist_images is not None:
            wrist_images = _normalize_images(wrist_images)
        if extra_view_images is not None:
            extra_view_images = _normalize_images(extra_view_images)

    length = min(
        states.shape[0],
        actions.shape[0],
        rewards.shape[0],
        main_images.shape[0],
    )
    return EpisodeArrays(
        states=states[:length],
        actions=actions[:length],
        rewards=rewards[:length],
        human_flags=human_flags[:length],
        human_actions=human_actions[:length] if human_actions is not None else None,
        main_images=main_images[:length],
        wrist_images=wrist_images[:length] if wrist_images is not None else None,
        extra_view_images=extra_view_images[:length]
        if extra_view_images is not None
        else None,
    )


def build_rlt_feature_model() -> torch.nn.Module:
    cfg = OmegaConf.create(RLT_FEATURE_MODEL_CFG)
    model = get_model(cfg)
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
    wrist_images = None
    if left_wrist is not None and right_wrist is not None:
        # AlohaInputs expects one wrist tensor per sample with shape [2,H,W,C]:
        # index 0 is left wrist, index 1 is right wrist.
        wrist_images = torch.stack([left_wrist, right_wrist], dim=1)
    elif left_wrist is not None:
        wrist_images = left_wrist[:, None]
    elif right_wrist is not None:
        wrist_images = right_wrist[:, None]

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
    return {key: torch.cat(values, dim=0) for key, values in chunks.items()}


def _pad_take(array: np.ndarray, start: int, length: int) -> np.ndarray:
    end = start + length
    if end <= array.shape[0]:
        return array[start:end]
    pad_count = end - array.shape[0]
    tail = array[start:]
    pad = np.repeat(array[-1:], pad_count, axis=0)
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
        reward_chunk = _pad_take(episode.rewards[:, None], start, ACTION_CHUNK)[:, 0]
        human_flag_chunk = _pad_take(
            episode.human_flags[:, None].astype(np.bool_), start, ACTION_CHUNK
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
            terminations[-1] = True
            dones[-1] = True

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


def convert_episode(
    *,
    feature_model: torch.nn.Module,
    hdf5_path: str,
) -> Trajectory | None:
    episode = load_hdf5_episode(hdf5_path)
    if episode.actions.shape[0] < 2:
        return None

    max_start = max(0, episode.actions.shape[0] - 1)
    starts = np.arange(0, max_start, ACTION_STRIDE, dtype=np.int64)
    if starts.size == 0:
        starts = np.asarray([0], dtype=np.int64)
    next_indices = np.minimum(starts + ACTION_CHUNK, episode.actions.shape[0] - 1)

    curr_features = extract_rlt_obs_for_indices(feature_model, episode, starts)
    next_features = extract_rlt_obs_for_indices(feature_model, episode, next_indices)
    model_weights_id = uuid.uuid5(uuid.NAMESPACE_URL, hdf5_path).hex[:12]

    return build_chunked_trajectory(
        episode=episode,
        curr_features=curr_features,
        next_features=next_features,
        starts=starts,
        model_weights_id=model_weights_id,
    )


def convert_directory() -> None:
    hdf5_paths = sorted(glob.glob(os.path.join(HDF5_DIR, "episode_*.hdf5")))
    if not hdf5_paths:
        raise FileNotFoundError(f"No episode_*.hdf5 files found under {HDF5_DIR}")

    total_episodes = len(hdf5_paths)

    if MAX_EPISODES > 0:
        hdf5_paths = hdf5_paths[:MAX_EPISODES]

    print(
        f"Found {total_episodes} episodes; "
        f"converting {len(hdf5_paths)} episodes."
    )

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
        trajectory = convert_episode(feature_model=feature_model, hdf5_path=hdf5_path)
        if trajectory is None:
            continue
        replay_buffer.add_trajectories([trajectory])

    output_path = Path(OUTPUT_BUFFER_DIR)
    output_path.mkdir(parents=True, exist_ok=True)
    replay_buffer.save_checkpoint(str(output_path))
    replay_buffer.close()
    print(f"Saved RLinf replay buffer to {output_path}")
    print(f"Stats: {replay_buffer.get_stats()}")


if __name__ == "__main__":
    convert_directory()
