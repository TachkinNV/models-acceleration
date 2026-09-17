"""Бенчмарк TensorRT engine (TensorRT 8.x + pycuda)."""

import time


def benchmark_tensorrt(config) -> dict:
    """Прогоняет готовый TensorRT engine на разных batch size.

    Требует tensorrt и pycuda. CUDA-контекст создаётся pycuda и
    корректно освобождается после бенчмарка.
    """
    import numpy as np
    import pycuda.driver as cuda
    import tensorrt as trt

    engine_path = config.engine_path
    if not engine_path.exists():
        raise FileNotFoundError(f"TensorRT engine не найден: {engine_path}")

    cuda.init()
    device = cuda.Device(0)
    cuda_context = device.make_context()
    print(f"CUDA контекст создан для {device.name()}")

    print(f"\n{'=' * 60}")
    print("Бенчмарк: TensorRT")
    print(f"{'=' * 60}")

    results = {}
    engine = None
    try:
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        with open(engine_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        if engine is None:
            raise RuntimeError(f"Не удалось десериализовать engine: {engine_path}")

        input_idx = engine.get_binding_index("input")
        output_idx = engine.get_binding_index("output")
        if input_idx < 0 or output_idx < 0:
            names = [engine.get_binding_name(i) for i in range(engine.num_bindings)]
            raise RuntimeError(f"Байндинги input/output не найдены. Доступны: {names}")

        input_binding_shape = tuple(engine.get_binding_shape(input_idx))
        output_binding_shape = tuple(engine.get_binding_shape(output_idx))
        num_classes = output_binding_shape[1] if output_binding_shape[1] > 0 else config.num_classes

        for batch_size in config.batch_sizes:
            print(f"\nТестируем batch size = {batch_size}...")
            context = engine.create_execution_context()

            try:
                if input_binding_shape[0] == -1:
                    ok = context.set_binding_shape(
                        input_idx, (batch_size,) + input_binding_shape[1:]
                    )
                    if not ok:
                        raise RuntimeError(
                            f"Не удалось задать batch size {batch_size} "
                            f"(профиль engine: {input_binding_shape})"
                        )

                input_data = np.random.randn(batch_size, *config.input_shape).astype(np.float32)
                output_data = np.empty((batch_size, num_classes), dtype=np.float32)

                d_input = cuda.mem_alloc(input_data.nbytes)
                d_output = cuda.mem_alloc(output_data.nbytes)
                cuda.memcpy_htod(d_input, input_data.ravel())

                bindings = [0] * engine.num_bindings
                bindings[input_idx] = int(d_input)
                bindings[output_idx] = int(d_output)

                for _ in range(10):
                    context.execute_v2(bindings)
                cuda.Context.synchronize()

                start = time.perf_counter()
                for _ in range(config.benchmark_iterations):
                    context.execute_v2(bindings)
                cuda.Context.synchronize()
                elapsed = time.perf_counter() - start

                fps = (config.benchmark_iterations * batch_size) / elapsed
                latency_ms = (elapsed / config.benchmark_iterations) * 1000
                results[batch_size] = {"fps": fps, "latency_ms": latency_ms}
                print(f"  Batch Size {batch_size:3d}: {fps:7.2f} img/sec")
            finally:
                del context
                d_input.free()
                d_output.free()
    finally:
        engine = None
        cuda_context.pop()

    return results
