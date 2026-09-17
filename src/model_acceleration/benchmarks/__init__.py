"""Бенчмарки методов ускорения инференса."""

from .onnx_benchmark import benchmark_onnx_runtime
from .torch_benchmarks import benchmark_amp, benchmark_baseline, benchmark_inference, benchmark_torch_compile
from .trt_benchmark import benchmark_tensorrt

__all__ = [
    "benchmark_amp",
    "benchmark_baseline",
    "benchmark_inference",
    "benchmark_onnx_runtime",
    "benchmark_tensorrt",
    "benchmark_torch_compile",
]
