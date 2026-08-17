"""Compute normalization statistics for a config.

This script is used to compute the normalization statistics for a given config. It
will compute the mean and standard deviation of the data in the dataset and save it
to the config assets directory.
"""

import pathlib

import numpy as np
import tqdm
import tyro

import openpi.models.model as _model
import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as transforms


class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


def norm_stats_output_paths(config: _config.TrainConfig, data_config: _config.DataConfig) -> list[pathlib.Path]:
    if data_config.asset_id is None:
        raise ValueError("Data config must have an asset_id")

    output_paths = [config.assets_dirs / data_config.asset_id]
    if data_config.repo_id is None or data_config.repo_id == data_config.asset_id:
        return output_paths

    repo_path = pathlib.Path(data_config.repo_id)
    if repo_path.is_absolute():
        return output_paths

    compatibility_path = config.assets_dirs / repo_path
    if compatibility_path not in output_paths:
        output_paths.append(compatibility_path)
    return output_paths


def save_norm_stats(
    config: _config.TrainConfig,
    data_config: _config.DataConfig,
    norm_stats: dict[str, normalize.NormStats],
) -> None:
    for output_path in norm_stats_output_paths(config, data_config):
        print(f"Writing stats to: {output_path}")
        normalize.save(output_path, norm_stats)


def _input_transforms_for_norm_stats(data_config: _config.DataConfig) -> list[transforms.DataTransformFn]:
    # Norm stats only use state/actions, but short-memory datasets still carry multi-timestamp
    # images/states. Split those before ALOHA transforms see a 4D image tensor.
    temporal_transforms = []
    if data_config.num_video_frames > 1:
        temporal_transforms.append(transforms.SplitTemporalFrames(num_video_frames=data_config.num_video_frames))

    return [
        *data_config.repack_transforms.inputs,
        *temporal_transforms,
        *data_config.data_transforms.inputs,
        # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
        RemoveStrings(),
    ]


def create_torch_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    model_config: _model.BaseModelConfig,
    num_workers: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")
    dataset = _data_loader.create_torch_dataset(data_config, action_horizon, model_config)
    dataset = _data_loader.TransformedDataset(
        dataset,
        _input_transforms_for_norm_stats(data_config),
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
        shuffle = True
    else:
        num_batches = len(dataset) // batch_size
        shuffle = False
    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def create_rlds_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    dataset = _data_loader.create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=False)
    dataset = _data_loader.IterableTransformedDataset(
        dataset,
        _input_transforms_for_norm_stats(data_config),
        is_batched=True,
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
    else:
        # NOTE: this length is currently hard-coded for DROID.
        num_batches = len(dataset) // batch_size
    data_loader = _data_loader.RLDSDataLoader(
        dataset,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def create_multi_torch_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    model_config: _model.BaseModelConfig,
    num_workers: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    """Create a combined data loader from multiple repo IDs for norm stats computation."""
    dataset = _data_loader.create_multi_torch_dataset(data_config, action_horizon, model_config)
    dataset = _data_loader.TransformedDataset(
        dataset,
        _input_transforms_for_norm_stats(data_config),
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
        shuffle = True
    else:
        num_batches = len(dataset) // batch_size
        shuffle = False
    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def main(config_name: str, max_frames: int | None = None):
    config = _config.get_config(config_name)
    data_config = config.data.create(config.assets_dirs, config.model)

    if data_config.multi_repo_ids:
        print(f"Computing norm stats for multi-task config: {config_name}")
        print(f"  Datasets: {data_config.multi_repo_ids}")
        data_loader, num_batches = create_multi_torch_dataloader(
            data_config, config.model.action_horizon, config.batch_size, config.model, config.num_workers, max_frames
        )
    elif data_config.rlds_data_dir is not None:
        data_loader, num_batches = create_rlds_dataloader(
            data_config, config.model.action_horizon, config.batch_size, max_frames
        )
    else:
        data_loader, num_batches = create_torch_dataloader(
            data_config, config.model.action_horizon, config.batch_size, config.model, config.num_workers, max_frames
        )

    keys = ["state", "actions"]
    stats = {key: normalize.RunningStats() for key in keys}

    for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats"):
        for key in keys:
            stats[key].update(np.asarray(batch[key]))

    norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}

    save_norm_stats(config, data_config, norm_stats)


if __name__ == "__main__":
    tyro.cli(main)
