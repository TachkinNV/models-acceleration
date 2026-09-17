"""Датасеты: CIFAR-10 и Imagenette2, препроцессинг, скачивание."""

import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .config import DATASETS, DATA_DIR, IMAGENETTE_URL

CIFAR10_MEAN = DATASETS["cifar10"]["mean"]
CIFAR10_STD = DATASETS["cifar10"]["std"]


def imagenette_root(root=DATA_DIR) -> Path:
    return Path(root) / "imagenette2"


def download_imagenette(root=DATA_DIR) -> Path:
    """Скачивает и распаковывает Imagenette2 (полный размер, ~1.5 ГБ)."""
    root = Path(root)
    target = imagenette_root(root)
    if (target / "train").exists():
        print(f"Imagenette2 уже распакована: {target}")
        return target

    tgz = root / "imagenette2.tgz"
    if not tgz.exists():
        root.mkdir(parents=True, exist_ok=True)
        print(f"Скачиваю Imagenette2 из {IMAGENETTE_URL} (~1.5 ГБ)...")
        urllib.request.urlretrieve(IMAGENETTE_URL, tgz)

    print("Распаковываю...")
    with tarfile.open(tgz, "r:gz") as f:
        f.extractall(root)
    print(f"Imagenette2 готова: {target}")
    return target


def get_train_transform(dataset: str, image_size: int = 224) -> transforms.Compose:
    stats = DATASETS[dataset]
    if dataset == "cifar10":
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(),
                transforms.Resize(image_size, antialias=True),
                transforms.Normalize(stats["mean"], stats["std"]),
            ]
        )
    if dataset == "imagenette":
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size, scale=(0.5, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(stats["mean"], stats["std"]),
            ]
        )
    raise ValueError(dataset)


def get_eval_transform(dataset: str, image_size: int = 224) -> transforms.Compose:
    stats = DATASETS[dataset]
    if dataset == "cifar10":
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(image_size, antialias=True),
                transforms.Normalize(stats["mean"], stats["std"]),
            ]
        )
    if dataset == "imagenette":
        return transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(stats["mean"], stats["std"]),
            ]
        )
    raise ValueError(dataset)


def create_train_dataset(dataset: str, root=DATA_DIR):
    if dataset == "cifar10":
        return datasets.CIFAR10(root=str(root), train=True, download=False,
                                transform=get_train_transform(dataset))
    if dataset == "imagenette":
        download_imagenette(root)
        return datasets.ImageFolder(str(imagenette_root(root) / "train"),
                                    transform=get_train_transform(dataset))
    raise ValueError(dataset)


def create_eval_dataset(dataset: str, root=DATA_DIR):
    """Тестовый (val) датасет с eval-препроцессингом."""
    if dataset == "cifar10":
        return datasets.CIFAR10(root=str(root), train=False, download=False,
                                transform=get_eval_transform(dataset))
    if dataset == "imagenette":
        download_imagenette(root)
        return datasets.ImageFolder(str(imagenette_root(root) / "val"),
                                    transform=get_eval_transform(dataset))
    raise ValueError(dataset)


def eval_batches(dataset: str, batch_size: int, num_workers: int = 8):
    """Итератор по тесту: (x, y) — препроцессенные батчи.

    x — CPU-тензор float32 (N, 3, 224, 224); одинаковые входы для всех методов.
    """
    ds = create_eval_dataset(dataset)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    for x, y in loader:
        yield x, y.numpy()


def calibration_batches(
    dataset: str,
    num_images: int = 512,
    batch_size: int = 32,
    num_workers: int = 8,
) -> list[np.ndarray]:
    """Батчи из train-сплита с eval-препроцессингом — для INT8-калибровки.

    Детерминированный shuffle (seed=42): калибровка воспроизводима.
    """
    if dataset == "cifar10":
        ds = datasets.CIFAR10(root=str(DATA_DIR), train=True, download=False,
                              transform=get_eval_transform(dataset))
    elif dataset == "imagenette":
        ds = datasets.ImageFolder(str(imagenette_root() / "train"),
                                  transform=get_eval_transform(dataset))
    else:
        raise ValueError(dataset)

    generator = torch.Generator().manual_seed(42)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=generator,
    )
    batches = []
    collected = 0
    for x, _ in loader:
        batches.append(x.numpy().astype(np.float32))
        collected += x.shape[0]
        if collected >= num_images:
            break
    return batches


def load_cifar10_test(root=DATA_DIR) -> tuple[torch.Tensor, np.ndarray]:
    """Сырой CIFAR-10 тест: uint8 (N, 32, 32, 3) и метки (для быстрого пути на GPU)."""
    ds = datasets.CIFAR10(root=str(root), train=False, download=False)
    images = torch.from_numpy(ds.data)
    labels = np.asarray(ds.targets)
    return images, labels


class BatchPreprocessor:
    """Препроцессинг CIFAR-10 на GPU: HWC uint8 -> CHW float32, resize, normalize."""

    def __init__(
        self,
        device: torch.device,
        image_size: int = 224,
        mean: tuple = CIFAR10_MEAN,
        std: tuple = CIFAR10_STD,
    ):
        self.device = device
        self.size = image_size
        self.mean = torch.tensor(mean, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(std, device=device).view(1, 3, 1, 1)

    def __call__(self, batch_uint8: torch.Tensor) -> torch.Tensor:
        x = batch_uint8.to(self.device, non_blocking=True)
        x = x.permute(0, 3, 1, 2).float().div_(255.0)
        x = F.interpolate(
            x, size=self.size, mode="bilinear", antialias=True, align_corners=False
        )
        return (x - self.mean) / self.std
