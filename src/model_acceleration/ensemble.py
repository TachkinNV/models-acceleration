"""Ансамбли моделей: soft-voting, TTA, анализ ошибок, скоростной бенчмарк."""

import numpy as np
import torch
import torch.nn.functional as F

from .config import BenchmarkConfig
from .data import eval_batches
from .utils import get_device, load_model


def load_ensemble_models(archs: list, config: BenchmarkConfig):
    """Загружает список моделей (по одной на архитектуру) на одно устройство."""
    device = get_device()
    models = []
    for arch in archs:
        cfg = BenchmarkConfig(arch=arch, dataset=config.dataset, num_classes=config.num_classes)
        models.append(load_model(arch, cfg.resolve_weights(), config.num_classes, device))
    return models


def _collect_probs(forward, dataset: str, batch_size: int) -> np.ndarray:
    """forward: callable(x_cpu) -> logits (N, num_classes); возвращает softmax-вероятности."""
    probs_batches = []
    for x, _ in eval_batches(dataset, batch_size):
        logits = forward(x)
        p = F.softmax(torch.from_numpy(np.asarray(logits)), dim=1)
        probs_batches.append(p.numpy())
    return np.concatenate(probs_batches)


def evaluate_member(model, dataset: str, batch_size: int, tta: bool = False):
    """Вероятности одного члена ансамбля; TTA — усреднение с горизонтальным флипом."""
    device = next(model.parameters()).device

    def forward(x):
        with torch.no_grad():
            xt = x.to(device)
            logits = model(xt)
            if tta:
                logits = logits + model(torch.flip(xt, dims=[3]))
        return logits.cpu().numpy()

    return _collect_probs(forward, dataset, batch_size)


def labels_of(dataset: str, batch_size: int = 32) -> np.ndarray:
    ys = []
    for _, y in eval_batches(dataset, batch_size):
        ys.append(y)
    return np.concatenate(ys)


def accuracy_from_probs(probs: np.ndarray, labels: np.ndarray) -> float:
    return 100.0 * float((probs.argmax(1) == labels).mean())


def analyze_errors(probs_by_model: dict, labels: np.ndarray, weights: list | None = None) -> dict:
    """Ошибки отдельных моделей, их пересечения и итог взвешенного ансамбля."""
    probs_list = list(probs_by_model.values())
    w = np.array(weights, dtype=float) if weights is not None else np.ones(len(probs_list))
    w = w / w.sum()
    ensemble_probs = np.tensordot(w, np.stack(probs_list), axes=1)
    errors = {}
    for name, probs in probs_by_model.items():
        errs = set(np.where(probs.argmax(1) != labels)[0].tolist())
        errors[name] = errs

    ensemble_errors = set(np.where(ensemble_probs.argmax(1) != labels)[0].tolist())
    return {
        "per_model_errors": errors,
        "ensemble_errors": ensemble_errors,
        "ensemble_probs": ensemble_probs,
    }


def evaluate_ensemble(config: BenchmarkConfig, archs: list, tta: bool = False,
                      weights: list | None = None) -> dict:
    """Оценивает индивидуальные точности и ансамбль (soft-voting, опц. взвешенный)."""
    device = get_device()
    labels = labels_of(config.dataset, config.batch_size_eval)

    models = load_ensemble_models(archs, config)
    probs_by_model = {}
    accs = {}
    for arch, model in zip(archs, models):
        probs = evaluate_member(model, config.dataset, config.batch_size_eval, tta=tta)
        probs_by_model[arch] = probs
        accs[arch] = accuracy_from_probs(probs, labels)
        print(f"  {arch}{' + TTA' if tta else ''}: {accs[arch]:.2f}%")
        model.to("cpu")
        torch.cuda.empty_cache()

    analysis = analyze_errors(probs_by_model, labels, weights=weights)
    ens_acc = accuracy_from_probs(analysis["ensemble_probs"], labels)
    w_desc = f", веса {weights}" if weights is not None else ""
    print(f"  Ансамбль ({len(archs)} моделей){' + TTA' if tta else ''}{w_desc}: {ens_acc:.2f}%")

    return {
        "individual": accs,
        "ensemble": ens_acc,
        "labels": labels,
        "analysis": analysis,
    }


def print_ensemble_report(result: dict, labels: np.ndarray) -> None:
    """Таблица точности и анализ ошибок ансамбля."""
    accs = result["individual"]
    analysis = result["analysis"]
    total = len(labels)

    width = 28
    print(f"\n{'=' * (width + 14)}")
    print("АНСАМБЛЬ: ТОЧНОСТЬ (TOP-1)")
    print(f"{'=' * (width + 14)}\n")
    print(f"{'Модель':<{width}}{'Accuracy, %':>12}")
    for name, acc in accs.items():
        print(f"{name:<{width}}{acc:>12.2f}")
    print(f"{'—' * (width + 12)}")
    print(f"{'Soft-voting ансамбль':<{width}}{result['ensemble']:>12.2f}")

    print(f"\nАнализ ошибок (всего {total} изображений):")
    for name, errs in analysis["per_model_errors"].items():
        print(f"  {name}: {len(errs)} ошибок")
    common = set.intersection(*analysis["per_model_errors"].values()) \
        if analysis["per_model_errors"] else set()
    union = set.union(*analysis["per_model_errors"].values()) \
        if analysis["per_model_errors"] else set()
    print(f"  ошибаются все модели одновременно: {len(common)}")
    print(f"  ошибается хотя бы одна: {len(union)}")
    print(f"  итоговые ошибки ансамбля: {len(analysis['ensemble_errors'])}")


# ---------------------------------------------------------------------------
# Скоростной бенчмарк ансамбля
# ---------------------------------------------------------------------------

def _run_ensemble_batch(models, x, use_amp: bool) -> np.ndarray:
    """Последовательный прогон K моделей; усреднение softmax."""
    probs = None
    for model in models:
        with torch.no_grad():
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(x)
            else:
                logits = model(x)
        p = F.softmax(logits.float(), dim=1)
        probs = p if probs is None else probs + p
    return (probs / len(models)).cpu().numpy()


def benchmark_ensemble_torch(archs: list, config: BenchmarkConfig, use_amp: bool = False) -> dict:
    """Throughput ансамбля (последовательный инференс K моделей) в PyTorch."""
    from .utils import CUDATimer

    models = load_ensemble_models(archs, config)
    for m in models:
        m.eval()
    device = next(models[0].parameters()).device

    results = {}
    for bs in config.batch_sizes:
        dummy = torch.randn(bs, *config.input_shape, device=device)
        with torch.no_grad():
            for _ in range(config.warmup_iterations):
                _run_ensemble_batch(models, dummy, use_amp)
        torch.cuda.synchronize()

        timer = CUDATimer(device)
        with torch.no_grad():
            for _ in range(config.benchmark_iterations):
                _run_ensemble_batch(models, dummy, use_amp)
        elapsed = timer.elapsed()

        fps = (config.benchmark_iterations * bs) / elapsed
        latency = elapsed / config.benchmark_iterations * 1000
        results[bs] = {"fps": fps, "latency_ms": latency}
        print(f"  Batch Size {bs:3d}: {fps:7.2f} img/sec")
    return results


def benchmark_ensemble_trt(archs: list, config: BenchmarkConfig, fp16: bool = True) -> dict:
    """Throughput ансамбля в TensorRT: K engine'ов последовательно на общий батч."""
    import pycuda.driver as cuda
    import tensorrt as trt

    from .evaluation import ensure_engine

    # 1. Подготовка артефактов: ONNX-экспорт и сборка engine ДО pycuda-контекста
    #    (экспорт/сборка используют torch/TRT-builder, чужой контекст их ломает).
    engine_paths = []
    for arch in archs:
        cfg = BenchmarkConfig(
            arch=arch,
            dataset=config.dataset,
            num_classes=config.num_classes,
            input_shape=config.input_shape,
            batch_sizes=config.batch_sizes,
        )
        path = cfg.engine_variant_path("fp16") if fp16 else cfg.engine_path
        if not path.exists() and not cfg.onnx_path.exists():
            from .export import export_onnx

            model = load_model(arch, cfg.resolve_weights(), config.num_classes, get_device())
            export_onnx(model, cfg.onnx_path, config.input_shape)
        ensure_engine(cfg, fp16=fp16)
        engine_paths.append(path)

    cuda.init()
    cuda_context = cuda.Device(0).make_context()

    results = {}
    try:
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        engines = []
        for path in engine_paths:
            with open(path, "rb") as f:
                engine = runtime.deserialize_cuda_engine(f.read())
            if engine is None:
                raise RuntimeError(f"Не удалось десериализовать engine: {path}")
            engines.append(engine)

        # Персистентные контексты и буферы под max batch: от bs к bs меняется
        # только set_binding_shape (пересоздание контекстов в TRT 8.5 при
        # нескольких движках приводит к отказу set_binding_shape).
        max_bs = max(config.batch_sizes)
        states = []
        for engine in engines:
            context = engine.create_execution_context()
            input_idx = engine.get_binding_index("input")
            output_idx = engine.get_binding_index("output")
            input_shape = tuple(engine.get_binding_shape(input_idx))
            out_shape = tuple(engine.get_binding_shape(output_idx))
            num_classes = out_shape[1] if out_shape[1] > 0 else config.num_classes
            d_in = cuda.mem_alloc(max_bs * 4 * input_shape[1] * input_shape[2] * input_shape[3])
            d_out = cuda.mem_alloc(max_bs * num_classes * 4)
            bindings = [0] * engine.num_bindings
            bindings[input_idx] = int(d_in)
            bindings[output_idx] = int(d_out)
            states.append(
                {
                    "engine": engine,
                    "context": context,
                    "input_idx": input_idx,
                    "input_shape": input_shape,
                    "bindings": bindings,
                    "d_in": d_in,
                    "d_out": d_out,
                    "num_classes": num_classes,
                }
            )

        import time

        for bs in config.batch_sizes:
            x = np.random.randn(bs, *config.input_shape).astype(np.float32)

            for st in states:
                if st["input_shape"][0] == -1:
                    if not st["context"].set_binding_shape(
                        st["input_idx"], (bs,) + st["input_shape"][1:]
                    ):
                        raise RuntimeError(f"bs={bs} вне профиля engine")

            def run_once():
                for st in states:
                    cuda.memcpy_htod(st["d_in"], x)
                    st["context"].execute_v2(st["bindings"])

            for _ in range(config.warmup_iterations):
                run_once()
            cuda.Context.synchronize()

            start = time.perf_counter()
            for _ in range(config.benchmark_iterations):
                run_once()
            cuda.Context.synchronize()
            elapsed = time.perf_counter() - start

            fps = (config.benchmark_iterations * bs) / elapsed
            latency = elapsed / config.benchmark_iterations * 1000
            results[bs] = {"fps": fps, "latency_ms": latency}
            print(f"  Batch Size {bs:3d}: {fps:7.2f} img/sec")

        states = None
        engines = None
    finally:
        cuda_context.pop()

    return results
