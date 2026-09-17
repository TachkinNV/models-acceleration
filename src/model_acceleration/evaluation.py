"""Оценка точности (Top-1) методов инференса на тестовом сплите."""

import numpy as np
import torch

from .config import BenchmarkConfig
from .data import eval_batches
from .utils import get_device, load_model


def _evaluate(forward, dataset: str, batch_size: int, desc: str) -> float:
    """forward: callable(x_cpu_tensor) -> numpy-массив логитов (N, num_classes)."""
    correct = 0
    total = 0
    for x, target in eval_batches(dataset, batch_size):
        logits = forward(x)
        pred = np.asarray(logits).argmax(axis=1)
        correct += int((pred == target).sum())
        total += len(target)
    acc = 100.0 * correct / total
    print(f"  {desc}: {acc:.2f}%")
    return acc


def evaluate_torch(model, dataset, batch_size, desc) -> float:
    device = next(model.parameters()).device

    def forward(x):
        with torch.no_grad():
            return model(x.to(device)).cpu().numpy()

    return _evaluate(forward, dataset, batch_size, desc)


def evaluate_amp(model, dataset, batch_size, desc) -> float:
    device = next(model.parameters()).device
    dtype = torch.float16 if device.type == "cuda" else torch.bfloat16

    def forward(x):
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=dtype):
            return model(x.to(device)).float().cpu().numpy()

    return _evaluate(forward, dataset, batch_size, desc)


def evaluate_onnx(onnx_path, dataset, batch_size, desc) -> float:
    import onnxruntime as ort

    session = ort.InferenceSession(
        str(onnx_path),
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    input_name = session.get_inputs()[0].name

    def forward(x):
        return session.run(None, {input_name: x.numpy()})[0]

    return _evaluate(forward, dataset, batch_size, desc)


def evaluate_tensorrt(engine_path, dataset, batch_size, desc) -> float:
    import pycuda.driver as cuda
    import tensorrt as trt

    cuda.init()
    device = cuda.Device(0)
    cuda_context = device.make_context()

    try:
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        with open(engine_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        if engine is None:
            raise RuntimeError(f"Не удалось десериализовать engine: {engine_path}")

        input_idx = engine.get_binding_index("input")
        output_idx = engine.get_binding_index("output")
        num_classes = tuple(engine.get_binding_shape(output_idx))[1]
        input_shape = tuple(engine.get_binding_shape(input_idx))

        context = engine.create_execution_context()
        d_input = None
        d_output = None
        input_capacity = 0
        output_capacity = 0

        def forward(x):
            nonlocal d_input, d_output, input_capacity, output_capacity
            arr = np.ascontiguousarray(x.numpy())
            bs = arr.shape[0]
            if input_shape[0] == -1:
                if not context.set_binding_shape(input_idx, (bs,) + input_shape[1:]):
                    raise RuntimeError(f"batch size {bs} вне профиля engine")

            if input_capacity < arr.nbytes:
                if d_input is not None:
                    d_input.free()
                d_input = cuda.mem_alloc(arr.nbytes)
                input_capacity = arr.nbytes
            out_bytes = bs * num_classes * 4
            if output_capacity < out_bytes:
                if d_output is not None:
                    d_output.free()
                d_output = cuda.mem_alloc(out_bytes)
                output_capacity = out_bytes

            cuda.memcpy_htod(d_input, arr)
            bindings = [0] * engine.num_bindings
            bindings[input_idx] = int(d_input)
            bindings[output_idx] = int(d_output)
            context.execute_v2(bindings)
            out = np.empty((bs, num_classes), dtype=np.float32)
            cuda.memcpy_dtoh(out, d_output)
            return out

        acc = _evaluate(forward, dataset, batch_size, desc)
    finally:
        context = None
        engine = None
        cuda_context.pop()

    return acc


def evaluate_accuracy(config: BenchmarkConfig, methods: list) -> dict:
    """Оценивает Top-1 accuracy выбранных методов. Возвращает {метод: точность %}."""
    device = get_device()
    print(f"Тестовый сплит: {config.dataset}\n")

    results = {}
    model = None

    if any(m in methods for m in ("baseline", "compile", "amp")):
        model = load_model(config.arch, config.resolve_weights(), config.num_classes, device)

    if "baseline" in methods:
        results["Baseline (PyTorch)"] = evaluate_torch(
            model, config.dataset, config.batch_size_eval, "Baseline (PyTorch FP32)"
        )

    if "compile" in methods:
        compiled = torch.compile(model)
        results["torch.compile"] = evaluate_torch(
            compiled, config.dataset, config.batch_size_eval, "torch.compile"
        )

    if "amp" in methods:
        results["AMP (FP16)"] = evaluate_amp(
            model, config.dataset, config.batch_size_eval, "AMP (FP16)"
        )

    if "onnx" in methods:
        results["ONNX Runtime"] = evaluate_onnx(
            config.onnx_path, config.dataset, config.batch_size_eval, "ONNX Runtime (FP32)"
        )

    if "tensorrt" in methods:
        ensure_engine(config, fp16=False)
        results["TensorRT (FP32)"] = evaluate_tensorrt(
            config.engine_path, config.dataset, config.batch_size_eval, "TensorRT (FP32)"
        )

    if "tensorrt-fp16" in methods:
        ensure_engine(config, fp16=True)
        results["TensorRT (FP16)"] = evaluate_tensorrt(
            config.engine_variant_path("fp16"), config.dataset,
            config.batch_size_eval, "TensorRT (FP16)",
        )

    if "tensorrt-int8" in methods:
        ensure_int8_engine(config)
        results["TensorRT (INT8)"] = evaluate_tensorrt(
            config.engine_variant_path("int8"), config.dataset,
            config.batch_size_eval, "TensorRT (INT8)",
        )

    return results


def ensure_engine(config: BenchmarkConfig, fp16: bool) -> None:
    """Гарантирует наличие FP32/FP16 engine (при отсутствии собирает из ONNX)."""
    if fp16:
        path = config.engine_variant_path("fp16")
    else:
        path = config.engine_path
    if not path.exists():
        from .export import build_engine

        print(f"\nEngine не найден ({path}), собираю...")
        build_engine(
            config.onnx_path,
            path,
            max_batch_size=max(config.batch_sizes),
            fp16=fp16,
        )


def ensure_int8_engine(config: BenchmarkConfig) -> None:
    """Гарантирует наличие INT8 engine (с калибровкой при первой сборке)."""
    if config.engine_variant_path("int8").exists():
        return
    from .quantization import build_int8_engine

    build_int8_engine(config)
