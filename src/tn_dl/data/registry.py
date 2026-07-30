from __future__ import annotations

from collections.abc import Callable

from torch.utils.data import Dataset

from tn_dl.config import DataConfig

DatasetFactory = Callable[[DataConfig, bool], Dataset]


def _build_mnist(config: DataConfig, train: bool) -> Dataset:
    from torchvision.datasets import MNIST
    from torchvision.transforms import ToTensor

    return MNIST(
        root=config.data_dir,
        train=train,
        transform=ToTensor(),
        download=config.download,
    )


def _build_fashion_mnist(config: DataConfig, train: bool) -> Dataset:
    from torchvision.datasets import FashionMNIST
    from torchvision.transforms import ToTensor

    return FashionMNIST(
        root=config.data_dir,
        train=train,
        transform=ToTensor(),
        download=config.download,
    )


DATASET_REGISTRY: dict[str, DatasetFactory] = {
    "fashion_mnist": _build_fashion_mnist,
    "mnist": _build_mnist,
}
