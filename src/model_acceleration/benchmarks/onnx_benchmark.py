"""Бенчмарк ONNX Runtime."""

import numpy as np
import torch

from ..config import BenchmarkConfig


def benchmark_onnx_runtime(config: BenchmarkConfig) -> dict:
    """Инференс ONNX-модели через ONNX Runtime (CUDA EP с фолбэком на CPU)."""
    import onnx
    import onnxruntime as ort

    onnx.checker.check_model(onnx.load(str(config.onnx_path)))
    session = ort.InferenceSession(
        str(config.onnx_path),
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    input_name = session.get_inputs()[0].name

    print(f"\n{'=' * 60}")
    print("Бенчмарк: ONNX Runtime")
    print(f"{'=' * 60}")
    print(f"Провайдеры: {', '.join(session.get_providers())}")

    results = {}
    for batch_size in config.batch_sizes:
        dummy_input = np.random.randn(batch_size, *config.input_shape).astype(np.float32)

        for _ in range(config.warmup_iterations):
            session.run(None, {input_name: dummy_input})

        if torch.cuda.is_available():
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(config.benchmark_iterations):
                session.run(None, {input_name: dummy_input})
            end.record()
            torch.cuda.synchronize()
            elapsed = start.elapsed_time(end) / 1000.0
        else:
            import time

            t0 = time.perf_counter()
            for _ in range(config.benchmark_iterations):
                session.run(None, {input_name: dummy_input})
            elapsed = time.perf_counter() - t0

        fps = (config.benchmark_iterations * batch_size) / elapsed
        latency_ms = (elapsed / config.benchmark_iterations) * 1000
        results[batch_size] = {"fps": fps, "latency_ms": latency_ms}
        print(f"  Batch Size {batch_size:3d}: {fps:7.2f} img/sec")

    return results
