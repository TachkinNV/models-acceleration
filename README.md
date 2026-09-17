# Model Acceleration

Эксперименты по ускорению инференса нейросетей на **Tesla V100**: сравнение
PyTorch eager, `torch.compile`, AMP (FP16), ONNX Runtime и TensorRT
(FP32/FP16/INT8) — скорость **и точность** каждого метода. Модели: ResNet-18/50,
ConvNeXt-Tiny. Датасеты: Imagenette2 (основной) и CIFAR-10.

## Результаты

### Ансамбль: ConvNeXt-T + ResNet-50 + ViT-B/16 (99.67%)

Три архитектуры, все — файнтюн ImageNet-pretrained на Imagenette2.
Оценка — soft-voting (усреднение softmax), поддержка TTA и весов:

```bash
model-acceleration --dataset imagenette --accuracy \
    --ensemble convnext_tiny resnet50 vit_b_16 --ensemble-weights 2 1 1
```

Точность (Top-1, val-сплит 3 925 изображений):

| Конфигурация                          | Accuracy, % | Ошибок |
|---------------------------------------|------------:|-------:|
| ConvNeXt-Tiny (лучшая одиночная)      |       99.59 |     16 |
| ResNet-50 finetune                    |       99.06 |     37 |
| ViT-B/16 finetune                     |       98.90 |     43 |
| ConvNeXt-Tiny + TTA (hflip)           |       99.64 |     14 |
| Ансамбль, равные веса                 |       99.59 |     16 |
| **Ансамбль, веса 2:1:1**              |   **99.67** |     13 |
| Ансамбль (2:1:1) + TTA                |       99.64 |     14 |

Выводы:

- **Наивный soft-voting не работает при неравных участниках**: равные веса
  дают те же 99.59%, что и лучшая одиночная модель — более слабые ResNet-50
  и ViT «переголосовывают» правильные ответы ConvNeXt. Взвешенный ансамбль
  (2:1:1) выжимает ещё +0.08 п.п.
- **Потолок — шумные метки Imagenette**: 10 из 16 ошибок ConvNeXT — общие
  для всех трёх моделей, и все три уверенно предсказывают один и тот же
  «неверный» класс (например, все трое говорят «класс 9» с пробой 0.89 при
  метке «класс 8»). Это битые разметки — их не исправить ансамблированием.
- TTA даёт чуть меньше, чем ансамбль, при 2-кратной (а не 3-кратной)
  стоимости инференса.

Скорость ансамбля (последовательный инференс 3 моделей на батч):

| Рантайм                | bs=1  | bs=8  | bs=16 | bs=32 | Ускорение (bs=32) |
|------------------------|------:|------:|------:|------:|------------------:|
| PyTorch FP32           |   100 |   189 |   198 |   202 |             1.00x |
| PyTorch AMP (FP16)     |   140 |   318 |   430 |   523 |             2.58x |
| **TensorRT FP16**      |   137 |   508 |   621 |   675 |           **3.33x** |

- Ансамбль стоит дорого: лучшая одиночная модель в TRT FP16 — 2 113 img/s,
  ансамбль из трёх — 675 img/s (3.1x медленнее). Ускорение возвращает
  большую часть: FP32-ансамбль 202 → TRT FP16 675 img/s.
- Практический вывод: **если точность 99.59% достаточна — одна ConvNeXt-T
  в TRT FP16 (2 113 img/s) доминирует ансамбль по скорости на порядок при
  разнице в точности 0.08 п.п.**

### ConvNeXt-Tiny, файнтюн ImageNet-pretrained (val acc **99.59%**)

Современная архитектура (2022, 28.6M параметров) + файнтюн pretrained-весов:
8 эпох, ~7 минут на V100 — против 88.87% у ResNet-50, обученной с нуля за 21 мин.

Скорость (img/sec, batch 1–32, среднее за 50 итераций):

| Метод                       | bs=1  | bs=8  | bs=16 | bs=32 | Ускорение (bs=32) |
|-----------------------------|------:|------:|------:|------:|------------------:|
| Baseline (PyTorch eager)    |   357 |   777 |   858 |   878 |             1.00x |
| torch.compile (default)     |   506 |   994 |  1036 |  1057 |             1.20x |
| AMP (FP16)                  |   445 |  1412 |  1783 |  2055 |             2.34x |
| ONNX Runtime (CUDA EP)      |   277 |   690 |   743 |   768 |             0.87x |
| TensorRT 8.5.1.7 (FP32)     |   255 |   740 |   817 |   860 |             0.98x |
| **TensorRT 8.5.1.7 (FP16)** |   332 |  1474 |  1825 |  2113 |           **2.41x** |
| TensorRT 8.5.1.7 (INT8)     |   324 |  1432 |  1800 |  2094 |             2.38x |

Точность (Top-1, val-сплит 3 925 изображений):

| Конфигурация                  | Accuracy, % | Δ, п.п. |
|-------------------------------|------------:|--------:|
| Baseline (PyTorch FP32)       |       99.59 |       — |
| torch.compile / AMP / ORT / TRT FP32 / TRT FP16 | 99.59 | 0.00 |
| TensorRT INT8 (+FP16 fallback)|       99.64 |   +0.05 |
| **TensorRT чистый INT8 (DP4A)** |   **98.85** | **−0.74** |

Выводы:

- **Точность побита: 99.59% против 88.87%** (+10.7 п.п.). Ключ — pretrained-фичи,
  а не размер модели: Imagenette — сабсет ImageNet, и файнтюн сходится за
  пару эпох (99.08% уже после первой).
- **Архитектура меняет картину ускорений**: у ConvNeXt много elementwise-операций
  и LayerNorm, вычисления менее «матричные», чем у ResNet — TRT FP16 даёт
  лишь **2.41x** (против 5.23x на ResNet-50), а torch.compile наоборот сильнее
  (до **1.42x** на bs=1 против 1.12x). Выбор метода ускорения зависит от
  архитектуры, а не только от GPU.
- **INT8-квантизация на pretrained-модели практически бесплатна**: чистый INT8
  теряет всего 0.74 п.п. (98.85%), хотя на переобученной from-scratch ResNet-50
  та же процедура роняла точность на 57 п.п. Причина — «спокойные» логиты
  (диапазон −0.8..+4.5 против ±105). Но на V100 INT8 всё равно не даёт
  выигрыша в скорости: 1339 img/s против 2113 у FP16 (DP4A медленнее
  тензорных ядер).
- Практический итог для V100: **TRT FP16 или AMP** — в зависимости от того,
  нужен ли рантайм TensorRT.

### ResNet-50 с нуля на Imagenette2 (val acc 88.87%)

Обучение с нуля (40 эпох, SGD), batch 1–32, img/sec:
| Метод                       | bs=1  | bs=8  | bs=16 | bs=32 | Ускорение (bs=32) |
|-----------------------------|------:|------:|------:|------:|------------------:|
| Baseline (PyTorch eager)    |   347 |   908 |  1056 |  1152 |             1.00x |
| torch.compile (default)     |   388 |  1123 |  1322 |  1448 |             1.26x |
| AMP (FP16)                  |   588 |  1881 |  2230 |  2484 |             2.16x |
| ONNX Runtime (CUDA EP)      |   324 |   893 |  1018 |  1100 |             0.96x |
| TensorRT 8.5.1.7 (FP32)     |   259 |  1222 |  1452 |  1622 |             1.41x |
| **TensorRT 8.5.1.7 (FP16)** |   549 |  3324 |  4719 |  6028 |           **5.23x** |
| TensorRT 8.5.1.7 (INT8)     |   537 |  3281 |  4684 |  6018 |             5.22x |

Top-1 accuracy на val-сплите (3 925 изображений):

| Метод              | Accuracy, % | Δ, п.п. | Скорость (bs=32) |
|--------------------|------------:|--------:|-----------------:|
| Baseline (PyTorch) |       88.87 |       — |         1152 img/s |
| torch.compile      |       88.87 |    0.00 |         1448 img/s |
| AMP (FP16)         |       88.89 |   +0.03 |         2484 img/s |
| ONNX Runtime       |       88.87 |    0.00 |         1100 img/s |
| TensorRT (FP32)    |       88.87 |    0.00 |         1622 img/s |
| TensorRT (FP16)    |       88.87 |    0.00 |         6028 img/s |
| TensorRT (INT8)    |       88.87 |    0.00 |         6018 img/s |

**Диагностика INT8** (см. подробности ниже): engine «INT8» с фолбэком FP16
на самом деле исполняет FP16-ядра — поэтому скорость и точность совпадают
с FP16. Чистый INT8 без фолбэка даёт другой результат:

| Конфигурация INT8            | Accuracy, % | Скорость (bs=32) |
|------------------------------|------------:|-----------------:|
| INT8 + FP16 fallback         |       88.87 |         6018 img/s |
| INT8 только (entropy-калибр.)|       32.08 |         5293 img/s (−13% к FP16) |
| INT8 только (MinMax-калибр.) |       11.29 |         — |

Выводы:

- **TensorRT FP16 — оптимум для V100: 5.23x без потери точности.**
- **INT8 на V100 бессмыслен**: при разрешённом FP16 builder выбирает FP16-ядра
  (INT8 на Volta идёт через DP4A и медленнее тензорных ядер FP16); чистый INT8
  ещё и **теряет 57 п.п. точности**. INT8-тензорные ядра появляются только
  с архитектуры Turing/Ampere — там и есть смысл пробовать INT8.
- **Катастрофический провал чистого INT8 — свойство модели, а не баг**: модель,
  обученная с нуля на маленьком датасете (9.5k изображений), переобучилась
  (train acc 98.8%) и выдаёт логиты с диапазоном до ±105. PTQ-квантизация
  сжимает их в ±4 и сеть вырождается. Подтверждено контрастом с ConvNeXt-Tiny
  c pretrained-весами: там чистый INT8 теряет всего 0.74 п.п. (см. раздел выше).
- **AMP (FP16)** — лучший вариант внутри PyTorch: 2.16x «бесплатно».
- **torch.compile** — стабильные +26%; **ONNX Runtime** ≈ baseline.

### CIFAR-10 (первый эксперимент)

ResNet-50 (FP32), batch 1–32, img/sec; точность всех методов — 84.81%:

| Метод                      | bs=1  | bs=8  | bs=16 | bs=32 | Ускорение (bs=32) |
|----------------------------|------:|------:|------:|------:|------------------:|
| Baseline (PyTorch eager)   |   373 |   938 |  1089 |  1160 |             1.00x |
| torch.compile (default)    |   395 |  1154 |  1371 |  1474 |             1.27x |
| AMP (FP16)                 |   608 |  1895 |  2247 |  2499 |             2.15x |
| ONNX Runtime (CUDA EP)     |   382 |   906 |  1043 |  1107 |             0.95x |
| TensorRT 8.5.1.7 (FP32)    |   259 |  1226 |  1456 |  1641 |             1.41x |
| TensorRT 8.5.1.7 (FP16)    |   476 |  3145 |  4696 |  6053 |             5.22x |

CIFAR-10 оказался слишком лёгким: логиты хорошо разделены, поэтому **все**
методы (включая FP16) дали идентичные предсказания на всех 10 000 тестовых
изображениях — сравнивать методы по точности неинформативно. Отсюда переход
на Imagenette2.

## Структура репозитория

```
├── src/model_acceleration/   # пакет бенчмарков
│   ├── config.py             #   датасеты, пути, BenchmarkConfig
│   ├── utils.py              #   загрузка моделей, выбор устройства, CUDATimer
│   ├── data.py               #   CIFAR-10/Imagenette2, скачивание, препроцессинг
│   ├── train.py              #   обучение: с нуля / файнтюн pretrained (CLI)
│   ├── export.py             #   экспорт ONNX, сборка TRT engine FP32/FP16
│   ├── quantization.py       #   INT8: entropy-калибровка + сборка engine
│   ├── evaluation.py         #   оценка Top-1 accuracy для всех методов
│   ├── ensemble.py           #   ансамбли: soft-voting, TTA, взвешивание, бенчмарк
│   ├── reporting.py          #   сводные таблицы (скорость, точность)
│   ├── benchmarks/
│   │   ├── torch_benchmarks.py   # baseline, torch.compile, AMP
│   │   ├── onnx_benchmark.py     # ONNX Runtime
│   │   └── trt_benchmark.py      # TensorRT (pycuda)
│   └── __main__.py           #   CLI (python -m model_acceleration)
├── models/                   # веса и артефакты — НЕ в git (см. models/README.md)
├── data/                     # датасеты — НЕ в git (см. data/README.md)
├── notebooks/                # исторические эксперименты (Jupyter)
├── torch2trt_from_git/       # клон torch2trt — НЕ в git
├── Dockerfile                # базовый образ CUDA 12.1
└── pyproject.toml
```

## Требования

| Компонент      | Версия | Комментарий |
|----------------|--------|-------------|
| GPU            | Tesla V100 (Volta, SM 7.0) | тензорные ядра FP16 |
| Python         | 3.10   | |
| PyTorch        | 2.7.1+cu118 | сборка под CUDA 11.8 |
| ONNX Runtime GPU | 1.23.2 | CUDA Execution Provider |
| TensorRT       | **8.5.1.7** | линейки 8.x — последние с поддержкой Volta; TRT 10+ на V100 не работает (`Unsupported SM: 0x700`) |
| pycuda         | 2022.2.2 | запуск TRT engine |
| cuDNN          | 9.x (torch/ORT) + 8.9 (TRT) | см. установку |

## Установка

Проверенная связка (полностью повторяет текущее окружение `torch_v100_pip`):

```bash
conda create -n model_acceleration python=3.10 -y
conda activate model_acceleration

# PyTorch (CUDA 11.8)
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu118

# ONNX + ONNX Runtime (GPU)
pip install onnx onnxruntime-gpu==1.23.2 numpy

# TensorRT 8.5.1.7 + pycuda
pip install tensorrt==8.5.1.7 pycuda==2022.2.2

# cuDNN 9 для ONNX Runtime (кладётся в $CONDA_PREFIX/lib)
conda install -c conda-forge "cudnn=9.3" -y

# cuDNN 8.9 для TensorRT (в тот же $CONDA_PREFIX/lib)
pip download nvidia-cudnn-cu11==8.9.7.29 --no-deps -d /tmp/cudnn8
unzip -j -o /tmp/cudnn8/nvidia_cudnn_cu11-*.whl "nvidia/cudnn/lib/*" -d $CONDA_PREFIX/lib/

# Пакет проекта
pip install -e .
```

TensorRT линкует `libcudnn.so.8`, PyTorch/ORT — `libcudnn.so.9`, поэтому перед
запуском бенчмарков:

```bash
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```

## Использование

```bash
conda activate model_acceleration   # или torch_v100_pip на этой машине
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Обучение: с нуля или файнтюн pretrained (Imagenette2 скачается сама)
model-acceleration-train --dataset imagenette --arch resnet50
model-acceleration-train --dataset imagenette --arch convnext_tiny --pretrained
model-acceleration-train --dataset imagenette --arch vit_b_16 --pretrained --batch-size 32

# Ансамбль: точность (индивидуальные + TTA + soft-voting с весами)
model-acceleration --dataset imagenette --accuracy \
    --ensemble convnext_tiny resnet50 vit_b_16 --ensemble-weights 2 1 1

# Ансамбль: скорость (FP32 vs AMP vs TensorRT FP16)
model-acceleration --dataset imagenette --ensemble-bench \
    --ensemble convnext_tiny resnet50 vit_b_16

# Полный прогон скорости (ONNX и engine собираются автоматически)
model-acceleration --dataset imagenette --arch convnext_tiny

# Оценка точности (Top-1)
model-acceleration --dataset imagenette --arch convnext_tiny --accuracy

# Все методы, включая FP16 и INT8 TensorRT
model-acceleration --dataset imagenette --methods baseline compile amp onnx tensorrt tensorrt-fp16 tensorrt-int8

# Выбор методов, архитектуры и batch sizes
model-acceleration --dataset cifar10 --methods baseline amp tensorrt
model-acceleration --dataset imagenette --arch resnet18 --batch-sizes 1 32 128
```

Все параметры: `model-acceleration --help`, `model-acceleration-train --help`.

Чистый INT8 без FP16-фолбэка (диагностика из таблиц выше) собирается так:

```python
from model_acceleration.config import BenchmarkConfig
from model_acceleration.quantization import build_int8_engine

config = BenchmarkConfig(arch="resnet50", dataset="imagenette")
build_int8_engine(config, fp16_fallback=False)
```

## Артефакты моделей

Подробности — в [models/README.md](models/README.md). Схема имён:
`models/{arch}_{dataset}[_baseline|пусто|_fp16|_int8].{pth,onnx,engine}`.
Например: `resnet50_imagenette_baseline.pth`, `resnet50_imagenette_fp16.engine`.

**Engine привязан к версии TensorRT и GPU** — при смене окружения пересоберите
(`--build-engine` / `--methods tensorrt-fp16 tensorrt-int8`). Первые engine'ы
собирались в докер-образе NVIDIA (TRT 8.6.0.12, CUDA 12.1) — такой способ
тоже работает, но требует той же версии TRT при запуске.

## Ноутбуки

`notebooks/` — история экспериментов: поиск рабочей связки версий
TensorRT/CUDA для V100, первые замеры. Не поддерживаются; актуальный код —
в `src/model_acceleration/`.
