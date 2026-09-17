"""INT8-квантизация TensorRT: entropy-калибровка и сборка engine."""

from pathlib import Path

import numpy as np


def build_int8_engine(config, num_calibration_images: int = 512,
                      fp16_fallback: bool = True) -> Path:
    """Собирает INT8 TensorRT engine (entropy-калибровка).

    fp16_fallback=True разрешает TRT выбирать FP16-ядра там, где они быстрее
    (на V100 INT8 идёт через DP4A и обычно медленнее FP16-тензорных ядер —
    builder с обоими флагами выбирает FP16). Чистый INT8: fp16_fallback=False.
    Калибровка идёт по train-сплиту с eval-препроцессингом — тест не используется.
    Кэш калибровки сохраняется рядом с engine и переиспользуется при пересборке.
    """
    import pycuda.driver as cuda
    import tensorrt as trt

    from .data import calibration_batches

    engine_path = config.engine_variant_path("int8")
    cache_path = engine_path.with_suffix(".cache")

    engine_path.parent.mkdir(parents=True, exist_ok=True)

    cuda.init()
    cuda_context = cuda.Device(0).make_context()

    class _EntropyCalibrator(trt.IInt8EntropyCalibrator2):
        def __init__(self, batches, cache_file):
            trt.IInt8EntropyCalibrator2.__init__(self)
            self.batches = batches
            self.cache_file = Path(cache_file)
            self.batch_idx = 0
            self.batch_size = int(batches[0].shape[0])
            self.device_input = cuda.mem_alloc(int(batches[0].nbytes))

        def get_batch_size(self):
            return self.batch_size

        def get_batch(self, names):
            if self.batch_idx >= len(self.batches):
                return None
            batch = np.ascontiguousarray(self.batches[self.batch_idx])
            cuda.memcpy_htod(self.device_input, batch)
            self.batch_idx += 1
            return [int(self.device_input)]

        def read_calibration_cache(self):
            if self.cache_file.exists():
                print(f"Кэш калибровки найден: {self.cache_file}")
                return self.cache_file.read_bytes()
            return None

        def write_calibration_cache(self, cache):
            self.cache_file.write_bytes(cache)
            print(f"Кэш калибровки сохранён: {self.cache_file}")

    try:
        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        parser = trt.OnnxParser(network, logger)

        with open(config.onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
                raise RuntimeError(
                    f"Не удалось распарсить {config.onnx_path}:\n" + "\n".join(errors)
                )

        builder_config = builder.create_builder_config()
        builder_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)

        if fp16_fallback and builder.platform_has_fast_fp16:
            builder_config.set_flag(trt.BuilderFlag.FP16)

        batches = calibration_batches(
            config.dataset, num_images=num_calibration_images,
            batch_size=config.batch_size_eval,
        )
        print(
            f"Калибровка INT8: {sum(b.shape[0] for b in batches)} изображений "
            f"из train-сплита ({config.dataset})"
        )
        builder_config.set_flag(trt.BuilderFlag.INT8)
        calibrator = _EntropyCalibrator(batches, cache_path)
        builder_config.int8_calibrator = calibrator

        input_tensor = network.get_input(0)
        channels, height, width = (int(d) for d in input_tensor.shape[1:])
        max_batch_size = max(config.batch_sizes)
        profile = builder.create_optimization_profile()
        profile.set_shape(
            input_tensor.name,
            (1, channels, height, width),
            (max_batch_size, channels, height, width),
            (max_batch_size, channels, height, width),
        )
        builder_config.add_optimization_profile(profile)
        builder_config.set_calibration_profile(profile)

        serialized = builder.build_serialized_network(network, builder_config)
        if serialized is None:
            raise RuntimeError("Сборка INT8 TensorRT engine не удалась")

        print(f"Калибровка: использовано батчей — {calibrator.batch_idx}")

        with open(engine_path, "wb") as f:
            f.write(serialized)
        print(f"INT8 TensorRT engine сохранён: {engine_path}")
    finally:
        cuda_context.pop()

    return engine_path
