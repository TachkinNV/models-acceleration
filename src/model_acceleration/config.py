"""Конфигурация: пути к артефактам, датасеты и параметры бенчмарков."""

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"

IMAGENETTE_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz"

DATASETS = {
    "cifar10": {
        "num_classes": 10,
        "mean": (0.4914, 0.4822, 0.4465),
        "std": (0.2470, 0.2435, 0.2616),
    },
    "imagenette": {
        "num_classes": 10,
        "mean": (0.485, 0.456, 0.406),  # ImageNet-статистики: Imagenette — сабсет ImageNet
        "std": (0.229, 0.224, 0.225),
    },
}


@dataclass
class BenchmarkConfig:
    arch: str = "resnet50"
    dataset: str = "cifar10"
    num_classes: int = 10
    input_shape: tuple = (3, 224, 224)
    batch_sizes: tuple = (1, 8, 16, 32)
    warmup_iterations: int = 5
    benchmark_iterations: int = 50
    batch_size_eval: int = 32
    compile_modes: tuple = ("default", "reduce-overhead", "max-autotune")
    weights_path: Path | None = None
    onnx_path: Path | None = None
    engine_path: Path | None = None

    def __post_init__(self):
        if self.dataset not in DATASETS:
            raise ValueError(f"Неизвестный датасет: {self.dataset!r}. Доступны: {sorted(DATASETS)}")
        if self.onnx_path is None:
            self.onnx_path = MODELS_DIR / f"{self.arch}_{self.dataset}.onnx"
        if self.engine_path is None:
            self.engine_path = MODELS_DIR / f"{self.arch}_{self.dataset}.engine"

    def resolve_weights(self) -> Path:
        if self.weights_path is not None:
            return self.weights_path
        return MODELS_DIR / f"{self.arch}_{self.dataset}_baseline.pth"

    def engine_variant_path(self, suffix: str) -> Path:
        """Путь к engine-варианту: 'fp16' / 'int8'."""
        p = self.engine_path
        return p.parent / f"{p.stem}_{suffix}{p.suffix}"
