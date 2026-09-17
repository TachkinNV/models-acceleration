"""Бенчмарки PyTorch: eager baseline, torch.compile, AMP."""

import torch
from torch import nn

from ..config import BenchmarkConfig
from ..utils import CUDATimer


def benchmark_inference(model: nn.Module, config: BenchmarkConfig, model_name: str) -> dict:
    """Замеряет throughput и latency модели для разных batch size."""
    device = next(model.parameters()).device

    print(f"\n{'=' * 60}")
    print(f"Бенчмарк: {model_name}")
    print(f"{'=' * 60}")

    results = {}
    for batch_size in config.batch_sizes:
        dummy_input = torch.randn(batch_size, *config.input_shape, device=device)

        with torch.no_grad():
            for _ in range(config.warmup_iterations):
                model(dummy_input)
        if device.type == "cuda":
            torch.cuda.synchronize()

        timer = CUDATimer(device)
        with torch.no_grad():
            for _ in range(config.benchmark_iterations):
                model(dummy_input)
        elapsed = timer.elapsed()

        fps = (config.benchmark_iterations * batch_size) / elapsed
        latency_ms = (elapsed / config.benchmark_iterations) * 1000
        results[batch_size] = {
            "fps": fps,
            "latency_ms": latency_ms,
            "time_seconds": elapsed,
        }
        print(
            f"  Batch Size {batch_size:3d}: {fps:7.2f} img/sec, "
            f"Latency: {latency_ms:5.2f} ms/batch"
        )

    return results


def benchmark_baseline(model: nn.Module, config: BenchmarkConfig) -> dict:
    """PyTorch eager без оптимизаций."""
    return benchmark_inference(model, config, "Baseline (PyTorch)")


def benchmark_torch_compile(
    model: nn.Module, config: BenchmarkConfig, modes: tuple | None = None
) -> dict:
    """torch.compile с разными режимами. Возвращает {название метода: результаты}."""
    device = next(model.parameters()).device
    warmup_input = torch.randn(max(config.batch_sizes), *config.input_shape, device=device)

    results = {}
    for mode in (modes or config.compile_modes):
        print(f"\n  Компиляция с режимом '{mode}'...")
        try:
            compiled_model = torch.compile(model, mode=mode)
            with torch.no_grad():
                for _ in range(10):
                    compiled_model(warmup_input)
            results[f"compile ({mode})"] = benchmark_inference(
                compiled_model, config, f"torch.compile ({mode})"
            )
        except Exception as e:
            print(f"  Ошибка с режимом {mode}: {e}")

    return results


class _AMPWrapper(nn.Module):
    """Обёртка для инференса в смешанной точности (FP16 на GPU / BF16 на CPU)."""

    def __init__(self, model: nn.Module, device: torch.device):
        super().__init__()
        self.model = model
        self.device = device
        self.dtype = torch.float16 if device.type == "cuda" else torch.bfloat16

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=self.device.type, dtype=self.dtype):
            return self.model(x)


def benchmark_amp(model: nn.Module, config: BenchmarkConfig) -> dict:
    """Automatic Mixed Precision (FP16)."""
    device = next(model.parameters()).device
    amp_model = _AMPWrapper(model, device).to(device)
    return benchmark_inference(amp_model, config, "AMP (FP16)")
