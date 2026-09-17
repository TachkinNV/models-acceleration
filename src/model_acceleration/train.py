"""Обучение baseline-модели: с нуля или файнтюн ImageNet-pretrained.

Примеры:
    model-acceleration-train --dataset imagenette --arch resnet50
    model-acceleration-train --dataset imagenette --arch convnext_tiny --pretrained
"""

import argparse
import math
import sys
import time

import torch
from torch import nn
from torch.utils.data import DataLoader

from .config import DATASETS, MODELS_DIR
from .data import create_eval_dataset, create_train_dataset
from .utils import ARCHS, create_model, get_device, print_device_info


def parse_args(argv: list | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="model-acceleration-train",
        description="Обучение baseline-модели (веса для бенчмарков).",
    )
    parser.add_argument("--dataset", default="imagenette", choices=sorted(DATASETS))
    parser.add_argument("--arch", default="resnet50", choices=sorted(ARCHS))
    parser.add_argument("--pretrained", action="store_true",
                        help="файнтюн ImageNet-pretrained весов (AdamW, мало эпох)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="по умолчанию: 40 с нуля / 8 при --pretrained")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=None,
                        help="peak LR (по умолчанию: 0.05 SGD с нуля / 1e-4 AdamW finetune)")
    parser.add_argument("--momentum", type=float, default=0.9, help="только для SGD")
    parser.add_argument("--weight-decay", type=float, default=None,
                        help="по умолчанию: 5e-4 с нуля / 0.05 при --pretrained")
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--warmup-epochs", type=int, default=None,
                        help="по умолчанию: 3 с нуля / 1 при --pretrained")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def lr_lambda(step: int, total_steps: int, warmup_steps: int) -> float:
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


@torch.no_grad()
def validate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        pred = model(x).argmax(1)
        correct += int((pred == y).sum())
        total += y.numel()
    model.train()
    return 100.0 * correct / total


def main(argv: list | None = None) -> int:
    args = parse_args(argv)
    torch.manual_seed(args.seed)

    device = get_device()
    print_device_info(device)

    epochs = args.epochs if args.epochs is not None else (8 if args.pretrained else 40)
    warmup_epochs = args.warmup_epochs if args.warmup_epochs is not None else (
        1 if args.pretrained else 3
    )

    num_classes = DATASETS[args.dataset]["num_classes"]
    train_ds = create_train_dataset(args.dataset)
    val_ds = create_eval_dataset(args.dataset)
    print(f"Датасет: {args.dataset} — train {len(train_ds)}, val {len(val_ds)}\n")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=max(2, args.workers // 2),
        pin_memory=True,
    )

    model = create_model(args.arch, num_classes=num_classes, pretrained=args.pretrained).to(device)
    model.train()

    if args.pretrained:
        lr = args.lr if args.lr is not None else 1e-4
        weight_decay = args.weight_decay if args.weight_decay is not None else 0.05
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        print(f"Файнтюн pretrained {args.arch}: AdamW, lr={lr}, wd={weight_decay}, "
              f"{epochs} эпох\n")
    else:
        lr = args.lr if args.lr is not None else 0.05
        weight_decay = args.weight_decay if args.weight_decay is not None else 5e-4
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=args.momentum,
            nesterov=True,
            weight_decay=weight_decay,
        )
        print(f"Обучение с нуля {args.arch}: SGD, lr={lr}, wd={weight_decay}, "
              f"{epochs} эпох\n")

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    total_steps = epochs * len(train_loader)
    warmup_steps = warmup_epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: lr_lambda(step, total_steps, warmup_steps),
    )

    weights_path = MODELS_DIR / f"{args.arch}_{args.dataset}_baseline.pth"
    best_acc = 0.0

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        running_loss, correct, seen = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item() * y.numel()
            correct += int((logits.argmax(1) == y).sum())
            seen += y.numel()

        val_acc = validate(model, val_loader, device)
        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"epoch {epoch:3d}/{epochs} | "
            f"loss {running_loss / seen:.4f} | train acc {100.0 * correct / seen:.2f}% | "
            f"val acc {val_acc:.2f}% | lr {lr_now:.4f} | {time.time() - t0:.1f}s",
            flush=True,
        )

        if val_acc > best_acc:
            best_acc = val_acc
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), weights_path)
            print(f"  -> лучший чекпоинт сохранён: {weights_path}")

    print(f"\nГотово. Лучший val acc: {best_acc:.2f}, веса: {weights_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
