from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from tn_dl.config import DataConfig
from tn_dl.data.registry import DATASET_REGISTRY


@dataclass(slots=True)
class DataLoaders:
    train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]]
    eval_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]]


def build_dataloaders(config: DataConfig) -> DataLoaders:
    try:
        dataset_factory = DATASET_REGISTRY[config.dataset]
    except KeyError as error:
        supported = ", ".join(sorted(DATASET_REGISTRY))
        raise ValueError(
            f"Unsupported dataset {config.dataset!r}. Available datasets: {supported}."
        ) from error

    train_dataset = _limit_dataset(dataset_factory(config, True), config.smoke_samples)
    eval_dataset = _limit_dataset(dataset_factory(config, False), config.smoke_samples)
    return DataLoaders(
        train_loader=DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
        ),
        eval_loader=DataLoader(
            eval_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
        ),
    )


def _limit_dataset(dataset: Dataset, sample_limit: int | None) -> Dataset:
    if sample_limit is None:
        return dataset
    if sample_limit < 1:
        raise ValueError("smoke_samples must be greater than zero when provided.")
    return Subset(dataset, range(min(sample_limit, len(dataset))))


__all__ = ["DataLoaders", "build_dataloaders"]
