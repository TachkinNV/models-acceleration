"""CLI для запуска бенчмарков.

Примеры:
    python -m model_acceleration
    python -m model_acceleration --dataset imagenette
    python -m model_acceleration --arch resnet18 --batch-sizes 1 32 128
    python -m model_acceleration --methods baseline amp onnx tensorrt tensorrt-int8
    python -m model_acceleration --accuracy --methods baseline amp tensorrt tensorrt-fp16 tensorrt-int8
    python -m model_acceleration --export-onnx --build-engine --fp16
"""

import argparse
import sys
from pathlib import Path

from .benchmarks.onnx_benchmark import benchmark_onnx_runtime
from .benchmarks.torch_benchmarks import benchmark_amp, benchmark_baseline, benchmark_torch_compile
from .benchmarks.trt_benchmark import benchmark_tensorrt
from .config import DATASETS, BenchmarkConfig
from .evaluation import ensure_engine, ensure_int8_engine, evaluate_accuracy
from .export import build_engine, export_onnx
from .reporting import compare_results, print_accuracy_table
from .utils import ARCHS, get_device, load_model, print_device_info

ALL_METHODS = (
    "baseline",
    "compile",
    "amp",
    "onnx",
    "tensorrt",
    "tensorrt-fp16",
    "tensorrt-int8",
)
DEFAULT_METHODS = ("baseline", "compile", "amp", "onnx", "tensorrt")


def parse_args(argv: list | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="model_acceleration",
        description="Бенчмарк методов ускорения инференса: PyTorch, torch.compile, "
        "AMP, ONNX Runtime, TensorRT (FP32/FP16/INT8).",
    )
    parser.add_argument("--dataset", default="cifar10", choices=sorted(DATASETS),
                        help="датасет (по умолчанию cifar10)")
    parser.add_argument("--arch", default="resnet50", choices=sorted(ARCHS),
                        help="архитектура модели (по умолчанию resnet50)")
    parser.add_argument("--weights", type=Path, default=None,
                        help="путь к .pth (по умолчанию models/<arch>_<dataset>_baseline.pth)")
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 16, 32])
    parser.add_argument("--warmup", type=int, default=5, help="итераций прогрева")
    parser.add_argument("--iterations", type=int, default=50, help="итераций замера")
    parser.add_argument("--methods", nargs="+", choices=ALL_METHODS,
                        default=list(DEFAULT_METHODS),
                        help="какие методы бенчмарить/оценивать")
    parser.add_argument("--eval-batch-size", type=int, default=32,
                        help="batch size при оценке точности")
    parser.add_argument("--accuracy", action="store_true",
                        help="режим оценки точности (Top-1) вместо бенчмарка скорости")
    parser.add_argument("--ensemble", nargs="+", default=None, metavar="ARCH",
                        help="ансамбль моделей (soft-voting) в режиме --accuracy; "
                             "например: --ensemble convnext_tiny resnet50 vit_b_16")
    parser.add_argument("--ensemble-weights", nargs="+", type=float, default=None, metavar="W",
                        help="веса моделей ансамбля (по порядку --ensemble), "
                             "например: --ensemble-weights 2 1 1")
    parser.add_argument("--ensemble-bench", action="store_true",
                        help="скоростной бенчмарк ансамбля: PyTorch FP32/AMP + TensorRT FP16")
    parser.add_argument("--compile-modes", nargs="+",
                        default=["default", "reduce-overhead", "max-autotune"],
                        help="режимы torch.compile")
    parser.add_argument("--onnx-path", type=Path, default=None)
    parser.add_argument("--engine-path", type=Path, default=None)
    parser.add_argument("--export-onnx", action="store_true",
                        help="экспортировать модель в ONNX перед бенчмарком")
    parser.add_argument("--build-engine", action="store_true",
                        help="собрать TensorRT engine из ONNX перед бенчмарком")
    parser.add_argument("--fp16", action="store_true",
                        help="использовать FP16 при сборке TensorRT engine")
    return parser.parse_args(argv)


def main(argv: list | None = None) -> int:
    args = parse_args(argv)

    config = BenchmarkConfig(
        arch=args.arch,
        dataset=args.dataset,
        num_classes=args.num_classes,
        batch_sizes=tuple(args.batch_sizes),
        warmup_iterations=args.warmup,
        benchmark_iterations=args.iterations,
        batch_size_eval=args.eval_batch_size,
        compile_modes=tuple(args.compile_modes),
        weights_path=args.weights,
    )
    if args.onnx_path:
        config.onnx_path = args.onnx_path
    if args.engine_path:
        config.engine_path = args.engine_path

    device = get_device()
    print_device_info(device)

    dataset_title = config.dataset.upper()

    if args.ensemble_bench:
        from .ensemble import benchmark_ensemble_torch, benchmark_ensemble_trt
        from .reporting import compare_results

        archs = args.ensemble or ["convnext_tiny", "resnet50", "vit_b_16"]
        print("\n" + "=" * 80)
        print(f"БЕНЧМАРК АНСАМБЛЯ {' + '.join(archs)} НА {dataset_title}")
        print("=" * 80)
        all_results = {}
        print(f"\nPyTorch FP32 (последовательный инференс {len(archs)} моделей):")
        all_results["Ensemble PyTorch FP32"] = benchmark_ensemble_torch(archs, config, use_amp=False)
        print(f"\nPyTorch AMP (FP16):")
        all_results["Ensemble PyTorch AMP"] = benchmark_ensemble_torch(archs, config, use_amp=True)
        print(f"\nTensorRT FP16:")
        all_results["Ensemble TensorRT FP16"] = benchmark_ensemble_trt(archs, config, fp16=True)
        compare_results(all_results, config.batch_sizes)
        print("\n" + "=" * 80)
        print("БЕНЧМАРК АНСАМБЛЯ ЗАВЕРШЕН")
        print("=" * 80)
        return 0

    if args.accuracy:
        print("\n" + "=" * 80)
        print(f"ОЦЕНКА ТОЧНОСТИ МЕТОДОВ ИНФЕРЕНСА {config.arch.upper()} НА {dataset_title}")
        print("=" * 80)
        if args.ensemble:
            from .ensemble import evaluate_ensemble, labels_of, print_ensemble_report
            from .ensemble import accuracy_from_probs, evaluate_member, load_ensemble_models

            archs = args.ensemble
            weights = args.ensemble_weights
            if weights is not None and len(weights) != len(archs):
                raise SystemExit("Число --ensemble-weights должно совпадать с числом моделей")
            print(f"Ансамбль: {' + '.join(archs)}\n")
            labels = labels_of(config.dataset, config.batch_size_eval)

            print("Индивидуальные модели:")
            result = evaluate_ensemble(config, archs, tta=False, weights=weights)

            print("\nС TTA (усреднение с горизонтальным флипом):")
            result_tta = evaluate_ensemble(config, archs, tta=True, weights=weights)

            print_ensemble_report(result, labels)

            width = 36
            print(f"\n{'=' * (width + 14)}")
            print("СВОДКА: ТОЧНОСТЬ ВСЕХ КОНФИГУРАЦИЙ")
            print(f"{'=' * (width + 14)}\n")
            print(f"{'Конфигурация':<{width}}{'Accuracy, %':>12}")
            for name, acc in result["individual"].items():
                print(f"{name:<{width}}{acc:>12.2f}")
            for name, acc in result_tta["individual"].items():
                print(f"{name + ' + TTA':<{width}}{acc:>12.2f}")
            print(f"{'—' * (width + 12)}")
            print(f"{'Ансамбль soft-voting':<{width}}{result['ensemble']:>12.2f}")
            print(f"{'Ансамбль + TTA':<{width}}{result_tta['ensemble']:>12.2f}")
            return 0

        if any(m in args.methods for m in ("onnx", "tensorrt", "tensorrt-fp16", "tensorrt-int8")) \
                and not config.onnx_path.exists():
            model = load_model(config.arch, config.resolve_weights(), config.num_classes, device)
            export_onnx(model, config.onnx_path, config.input_shape)
        results = evaluate_accuracy(config, args.methods)
        print_accuracy_table(results, config.dataset)
        return 0

    print("\n" + "=" * 80)
    print(f"БЕНЧМАРК МЕТОДОВ УСКОРЕНИЯ {config.arch.upper()} НА {dataset_title}")
    print("=" * 80)

    model = load_model(config.arch, config.resolve_weights(), config.num_classes, device)

    if args.export_onnx:
        export_onnx(model, config.onnx_path, config.input_shape)
    if args.build_engine:
        if not config.onnx_path.exists():
            export_onnx(model, config.onnx_path, config.input_shape)
        build_engine(config.onnx_path, config.engine_path,
                     max_batch_size=max(config.batch_sizes), fp16=args.fp16)

    all_results = {}

    if "baseline" in args.methods:
        all_results["Baseline (PyTorch)"] = benchmark_baseline(model, config)

    if "compile" in args.methods:
        all_results.update(benchmark_torch_compile(model, config))

    if "amp" in args.methods:
        all_results["AMP (FP16)"] = benchmark_amp(model, config)

    if "onnx" in args.methods:
        if not config.onnx_path.exists():
            print(f"\nONNX модель не найдена ({config.onnx_path}), выполняю экспорт...")
            export_onnx(model, config.onnx_path, config.input_shape)
        all_results["ONNX Runtime"] = benchmark_onnx_runtime(config)

    if "tensorrt" in args.methods:
        if not config.onnx_path.exists():
            export_onnx(model, config.onnx_path, config.input_shape)
        ensure_engine(config, fp16=False)
        all_results["TensorRT (FP32)"] = benchmark_tensorrt(config)

    if "tensorrt-fp16" in args.methods:
        if not config.onnx_path.exists():
            export_onnx(model, config.onnx_path, config.input_shape)
        ensure_engine(config, fp16=True)
        fp16_config = BenchmarkConfig(
            **{**config.__dict__, "engine_path": config.engine_variant_path("fp16")}
        )
        all_results["TensorRT (FP16)"] = benchmark_tensorrt(fp16_config)

    if "tensorrt-int8" in args.methods:
        if not config.onnx_path.exists():
            export_onnx(model, config.onnx_path, config.input_shape)
        ensure_int8_engine(config)
        int8_config = BenchmarkConfig(
            **{**config.__dict__, "engine_path": config.engine_variant_path("int8")}
        )
        all_results["TensorRT (INT8)"] = benchmark_tensorrt(int8_config)

    compare_results(all_results, config.batch_sizes)

    print("\n" + "=" * 80)
    print("БЕНЧМАРК ЗАВЕРШЕН")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
