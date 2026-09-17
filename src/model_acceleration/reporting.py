"""Табличный вывод результатов бенчмарков."""

BASELINE_KEY = "Baseline (PyTorch)"


def print_accuracy_table(results: dict, dataset: str = "") -> None:
    """Таблица Top-1 accuracy и отклонение от baseline."""
    if not results:
        return

    title = f"ТОЧНОСТЬ (TOP-1) НА ТЕСТЕ {dataset.upper()}" if dataset else "ТОЧНОСТЬ (TOP-1)"
    baseline = results.get(BASELINE_KEY)
    width = max(24, max(len(m) for m in results) + 2)

    print(f"\n{'=' * (width + 20)}")
    print(title)
    print(f"{'=' * (width + 20)}\n")

    print(f"{'Метод':<{width}}{'Accuracy, %':>12}{'Δ, п.п.':>10}")
    for method, acc in results.items():
        delta = f"{acc - baseline:+.2f}" if baseline is not None and method != BASELINE_KEY else "—"
        print(f"{method:<{width}}{acc:>12.2f}{delta:>10}")


def compare_results(all_results: dict, batch_sizes: tuple) -> None:
    """Печатает таблицу throughput (img/sec) и ускорение относительно baseline."""
    if not all_results:
        return

    methods = list(all_results)
    width = max(14, max(len(m) for m in methods) + 2)

    print(f"\n{'=' * (12 + width * len(methods))}")
    print("СРАВНЕНИЕ МЕТОДОВ УСКОРЕНИЯ (img/sec)")
    print(f"{'=' * (12 + width * len(methods))}\n")

    print(f"{'Batch Size':<12}" + "".join(f"{m:>{width}}" for m in methods))
    for bs in batch_sizes:
        row = f"{bs:<12}"
        for method in methods:
            res = all_results[method].get(bs)
            row += f"{res['fps']:>{width}.1f}" if res else f"{'N/A':>{width}}"
        print(row)

    if BASELINE_KEY not in all_results:
        return

    baseline = all_results[BASELINE_KEY]
    others = [m for m in methods if m != BASELINE_KEY]
    if not others:
        return

    print(f"\n{'=' * (12 + width * len(others))}")
    print("ОТНОСИТЕЛЬНОЕ УСКОРЕНИЕ (Baseline = 1.0x)")
    print(f"{'=' * (12 + width * len(others))}\n")

    print(f"{'Batch Size':<12}" + "".join(f"{m:>{width}}" for m in others))
    for bs in batch_sizes:
        if bs not in baseline:
            continue
        row = f"{bs:<12}"
        for method in others:
            res = all_results[method].get(bs)
            if res:
                row += f"{res['fps'] / baseline[bs]['fps']:>{width - 1}.2f}x"
            else:
                row += f"{'N/A':>{width}}"
        print(row)
