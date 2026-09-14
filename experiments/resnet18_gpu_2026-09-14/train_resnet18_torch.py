"""Train ResNet-18 on the PetImages split with an explicit CUDA/CPU backend.

The resulting best checkpoint is intended for later ONNX export and MindSpore
Lite conversion. The script aborts if CUDA is unavailable; it never falls back
to CPU during a requested GPU experiment.
"""

import argparse
import hashlib
import json
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


def parse_args():
    parser = argparse.ArgumentParser(description="CUDA ResNet-18 pet classifier")
    parser.add_argument("--dataset_path", required=True, help="PetImages directory containing train/ and eval/")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", choices=("GPU", "CPU"), default="GPU",
                        help="Execution backend; a requested GPU never falls back to CPU.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--pretrained", action="store_true", help="Use torchvision ImageNet weights if explicitly requested.")
    parser.add_argument("--max_train_steps", type=int, default=0,
                        help="Optional smoke-test limit; 0 consumes all training batches.")
    parser.add_argument("--max_eval_steps", type=int, default=0,
                        help="Optional smoke-test limit; 0 consumes all validation batches.")
    return parser.parse_args()


def set_determinism(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def evaluate(model, loader, criterion, device, max_steps):
    model.eval()
    loss_sum = 0.0
    correct = 0
    total = 0
    with torch.inference_mode():
        for step, (images, labels) in enumerate(loader, start=1):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss_sum += loss.item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
            if max_steps and step >= max_steps:
                break
    return loss_sum / total, correct / total


def peak_gpu_memory(device):
    return torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0


def main():
    args = parse_args()
    if args.device == "GPU" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Refusing to run a GPU experiment on CPU.")

    dataset_path = Path(args.dataset_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    if not (dataset_path / "train").is_dir() or not (dataset_path / "eval").is_dir():
        raise ValueError("dataset_path must contain train/ and eval/ directories.")
    output_dir.mkdir(parents=True, exist_ok=False)
    set_determinism(args.seed)

    device = torch.device("cuda:0" if args.device == "GPU" else "cpu")
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    train_set = datasets.ImageFolder(dataset_path / "train", transform=train_transform)
    eval_set = datasets.ImageFolder(dataset_path / "eval", transform=eval_transform)
    if train_set.classes != ["Cat", "Dog"] or eval_set.classes != ["Cat", "Dog"]:
        raise ValueError(f"Unexpected class order: train={train_set.classes}, eval={eval_set.classes}")
    loader_args = dict(batch_size=args.batch_size, num_workers=args.workers, pin_memory=device.type == "cuda")
    train_loader = DataLoader(train_set, shuffle=True, drop_last=True, **loader_args)
    eval_loader = DataLoader(eval_set, shuffle=False, drop_last=False, **loader_args)

    weights = models.ResNet18_Weights.IMAGENET1K_V1 if args.pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)

    environment = {
        "torch": torch.__version__,
        "torchvision": __import__("torchvision").__version__,
        "requested_device": args.device,
        "actual_device": str(device),
        "cuda_runtime": torch.version.cuda if device.type == "cuda" else None,
        "cuda_device": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "cuda_device_capability": torch.cuda.get_device_capability(device) if device.type == "cuda" else None,
        "python": sys.version,
        "platform": platform.platform(),
        "dataset_path": str(dataset_path),
        "classes": train_set.classes,
        "train_images": len(train_set),
        "eval_images": len(eval_set),
        "arguments": vars(args),
    }
    write_json(output_dir / "environment.json", environment)

    best_accuracy = -1.0
    best_path = None
    metrics_path = output_dir / "metrics.jsonl"
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    total_start = time.perf_counter()
    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        for epoch in range(1, args.epochs + 1):
            epoch_start = time.perf_counter()
            model.train()
            train_loss_sum = 0.0
            train_correct = 0
            train_total = 0
            for step, (images, labels) in enumerate(train_loader, start=1):
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                logits = model(images)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                train_loss_sum += loss.item() * labels.size(0)
                train_correct += (logits.argmax(dim=1) == labels).sum().item()
                train_total += labels.size(0)
                if args.max_train_steps and step >= args.max_train_steps:
                    break

            eval_loss, eval_accuracy = evaluate(model, eval_loader, criterion, device, args.max_eval_steps)
            record = {
                "epoch": epoch,
                "train_loss": train_loss_sum / train_total,
                "train_accuracy": train_correct / train_total,
                "validation_loss": eval_loss,
                "validation_accuracy": eval_accuracy,
                "epoch_seconds": round(time.perf_counter() - epoch_start, 4),
                "peak_gpu_memory_bytes": peak_gpu_memory(device),
            }
            if eval_accuracy >= best_accuracy:
                best_accuracy = eval_accuracy
                checkpoint_path = output_dir / f"resnet18_best_epoch{epoch:02d}.pth"
                torch.save({
                    "architecture": "resnet18",
                    "class_to_idx": train_set.class_to_idx,
                    "state_dict": model.state_dict(),
                    "epoch": epoch,
                    "validation_accuracy": eval_accuracy,
                    "pretrained": args.pretrained,
                }, checkpoint_path)
                best_path = checkpoint_path
                record["best_checkpoint_updated"] = True
                record["best_checkpoint"] = str(best_path)
            metrics_file.write(json.dumps(record) + "\n")
            metrics_file.flush()
            print(json.dumps(record), flush=True)

    summary = {
        "best_validation_accuracy": best_accuracy,
        "total_seconds": round(time.perf_counter() - total_start, 4),
        "peak_gpu_memory_bytes": peak_gpu_memory(device),
        "best_checkpoint": str(best_path),
        "best_checkpoint_sha256": sha256(best_path),
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
