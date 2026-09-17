"""Экспорт модели: PyTorch -> ONNX -> TensorRT engine."""

from pathlib import Path

import torch


def export_onnx(
    model: torch.nn.Module,
    onnx_path: Path | str,
    input_shape: tuple = (3, 224, 224),
    opset_version: int = 14,
) -> Path:
    """Экспортирует модель в ONNX с динамической осью batch.

    Важно: opset фиксируем на 14. Для ConvNeXt opset 17 экспортирует LayerNorm
    как ноду LayerNormalization, которую TensorRT 8.5 не парсит без плагина;
    на opset 14 LayerNorm раскладывается на примитивы и TRT собирает engine.
    """
    onnx_path = Path(onnx_path)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    device = next(model.parameters()).device
    dummy_input = torch.randn(1, *input_shape, device=device)

    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
        opset_version=opset_version,
    )
    print(f"ONNX модель сохранена: {onnx_path}")
    return onnx_path


def build_engine(
    onnx_path: Path | str,
    engine_path: Path | str,
    max_batch_size: int = 32,
    fp16: bool = False,
    workspace_gb: int = 1,
) -> Path:
    """Собирает TensorRT engine из ONNX-модели (TensorRT 8.x).

    Optimization profile настраивается на диапазон batch 1..max_batch_size.
    """
    import tensorrt as trt

    onnx_path = Path(onnx_path)
    engine_path = Path(engine_path)
    engine_path.parent.mkdir(parents=True, exist_ok=True)

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
            raise RuntimeError(f"Не удалось распарсить {onnx_path}:\n" + "\n".join(errors))

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb << 30)

    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("FP16 включён")

    input_tensor = network.get_input(0)
    channels, height, width = (int(d) for d in input_tensor.shape[1:])
    profile = builder.create_optimization_profile()
    profile.set_shape(
        input_tensor.name,
        (1, channels, height, width),
        (max_batch_size, channels, height, width),
        (max_batch_size, channels, height, width),
    )
    config.add_optimization_profile(profile)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError(f"Сборка TensorRT engine не удалась ({onnx_path})")

    with open(engine_path, "wb") as f:
        f.write(serialized)
    print(f"TensorRT engine сохранён: {engine_path}")
    return engine_path
