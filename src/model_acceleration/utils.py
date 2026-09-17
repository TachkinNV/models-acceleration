"""Общие утилиты: выбор устройства, загрузка модели, GPU-таймер."""

import time
from pathlib import Path

import torch
from torch import nn
from torchvision import models

ARCHS = {
    "resnet18": models.resnet18,
    "resnet50": models.resnet50,
    "convnext_tiny": models.convnext_tiny,
    "vit_b_16": models.vit_b_16,
}

# ImageNet-pretrained веса для файнтюна (None — только обучение с нуля)
PRETRAINED_WEIGHTS = {
    "resnet50": models.ResNet50_Weights.IMAGENET1K_V1,
    "convnext_tiny": models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1,
    "vit_b_16": models.ViT_B_16_Weights.IMAGENET1K_V1,
}


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def print_device_info(device: torch.device) -> None:
    print(f"Устройство: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA версия: {torch.version.cuda}")


def create_model(arch: str, num_classes: int = 10, pretrained: bool = False) -> nn.Module:
    """Создаёт модель; при pretrained загружает ImageNet-веса и заменяет голову."""
    if arch not in ARCHS:
        raise ValueError(f"Неизвестная архитектура: {arch!r}. Доступны: {sorted(ARCHS)}")

    if pretrained:
        if arch not in PRETRAINED_WEIGHTS:
            raise ValueError(f"Pretrained-веса для {arch!r} не настроены")
        model = ARCHS[arch](weights=PRETRAINED_WEIGHTS[arch])
        if arch.startswith("resnet"):
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif arch.startswith("convnext"):
            model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
        elif arch.startswith("vit"):
            model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
        else:
            raise ValueError(f"Не настроена замена головы для {arch!r}")
        return model

    return ARCHS[arch](weights=None, num_classes=num_classes)


def load_model(
    arch: str,
    weights_path: Path | None = None,
    num_classes: int = 10,
    device: torch.device | None = None,
) -> nn.Module:
    """Загружает модель с обученными весами."""
    device = device or get_device()
    model = create_model(arch, num_classes=num_classes)

    if weights_path is not None and Path(weights_path).exists():
        state_dict = torch.load(weights_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        print(f"Загружены веса из {weights_path}")
    else:
        print("Веса не найдены, используется случайная инициализация")

    return model.to(device).eval()


class CUDATimer:
    """Таймер на cuda.Event (GPU) или time.time() (CPU)."""

    def __init__(self, device: torch.device):
        self.device = device
        self.reset()

    def reset(self) -> None:
        if self.device.type == "cuda":
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)
            self.start_event.record()
        else:
            self.start_time = time.time()

    def elapsed(self) -> float:
        """Секунды с момента reset() (с синхронизацией GPU)."""
        if self.device.type == "cuda":
            self.end_event.record()
            torch.cuda.synchronize()
            return self.start_event.elapsed_time(self.end_event) / 1000.0
        return time.time() - self.start_time
