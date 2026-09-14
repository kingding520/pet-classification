"""Train ResNet-18 for the PetImages cat/dog dataset with MindSpore.

The script intentionally keeps GPU and CPU runs in one code path so their
results can be compared.  It does not silently fall back from GPU to CPU.
"""

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import mindspore as ms
from mindspore import Tensor, context, nn, ops, set_seed
from mindspore.common import dtype as mstype
from mindspore.train import Model
from mindspore.train.serialization import save_checkpoint
import mindspore.dataset as ds
import mindspore.dataset.transforms as transforms
import mindspore.dataset.vision as vision


class BasicBlock(nn.Cell):
    """The two-convolution residual block used by ResNet-18."""

    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, pad_mode="pad", padding=1, has_bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, pad_mode="pad", padding=1, has_bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.SequentialCell([
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, has_bias=False),
                nn.BatchNorm2d(out_channels),
            ])

    def construct(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(identity)
        return self.relu(out + identity)


class ResNet18(nn.Cell):
    """ResNet-18 classifier with two output categories."""

    def __init__(self, num_classes=2):
        super().__init__()
        self.in_channels = 64
        self.stem = nn.SequentialCell([
            nn.Conv2d(3, 64, kernel_size=7, stride=2, pad_mode="pad", padding=3, has_bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, pad_mode="same"),
        ])
        self.layer1 = self._make_layer(64, 2)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self.avg_pool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.flatten = nn.Flatten()
        self.fc = nn.Dense(512, num_classes)

    def _make_layer(self, out_channels, blocks, stride=1):
        layers = [BasicBlock(self.in_channels, out_channels, stride)]
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.in_channels, out_channels))
        return nn.SequentialCell(layers)

    def construct(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avg_pool(x)
        return self.fc(self.flatten(x))


def parse_args():
    parser = argparse.ArgumentParser(description="MindSpore ResNet-18 pet classifier")
    parser.add_argument("--dataset_path", required=True, help="PetImages directory containing train/ and eval/")
    parser.add_argument("--platform", choices=("GPU", "CPU"), required=True,
                        help="Target backend. GPU never falls back to CPU.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max_train_steps", type=int, default=0,
                        help="Optional one-run smoke limit; 0 uses all batches.")
    parser.add_argument("--max_eval_steps", type=int, default=0,
                        help="Optional one-run smoke limit; 0 uses all batches.")
    return parser.parse_args()


def create_dataset(root, training, batch_size, workers, max_steps):
    dataset = ds.ImageFolderDataset(str(root), shuffle=training, num_parallel_workers=workers)
    image_ops = [vision.Decode()]
    if training:
        image_ops.extend([vision.RandomResizedCrop((224, 224)), vision.RandomHorizontalFlip(prob=0.5)])
    else:
        image_ops.append(vision.Resize((224, 224)))
    image_ops.extend([
        vision.Rescale(1.0 / 255.0, 0.0),
        vision.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        vision.HWC2CHW(),
    ])
    dataset = dataset.map(operations=image_ops, input_columns="image", num_parallel_workers=workers)
    dataset = dataset.map(operations=transforms.TypeCast(mstype.int32), input_columns="label",
                          num_parallel_workers=workers)
    dataset = dataset.batch(batch_size, drop_remainder=training)
    if max_steps:
        dataset = dataset.take(max_steps)
    if dataset.get_dataset_size() == 0:
        raise ValueError("Dataset has no usable batches; check the path and batch size.")
    return dataset


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    args = parse_args()
    dataset_path = Path(args.dataset_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    if not (dataset_path / "train").is_dir() or not (dataset_path / "eval").is_dir():
        raise ValueError("dataset_path must contain train/ and eval/ directories.")

    set_seed(args.seed)
    context.set_context(mode=context.GRAPH_MODE, device_target=args.platform)
    actual_target = ms.get_context("device_target")
    if actual_target != args.platform:
        raise RuntimeError(f"Requested {args.platform}, but MindSpore selected {actual_target}; aborting.")

    train_ds = create_dataset(dataset_path / "train", True, args.batch_size, args.workers, args.max_train_steps)
    eval_ds = create_dataset(dataset_path / "eval", False, args.batch_size, args.workers, args.max_eval_steps)
    network = ResNet18(num_classes=2)
    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    optimizer = nn.Momentum(network.trainable_params(), learning_rate=args.lr,
                            momentum=0.9, weight_decay=1e-4)
    model = Model(network, loss_fn=loss, optimizer=optimizer, metrics={"accuracy"})

    environment = {
        "mindspore": ms.__version__,
        "python": sys.version,
        "platform": platform.platform(),
        "requested_device_target": args.platform,
        "actual_device_target": actual_target,
        "dataset_path": str(dataset_path),
        "train_steps": train_ds.get_dataset_size(),
        "eval_steps": eval_ds.get_dataset_size(),
        "arguments": vars(args),
    }
    write_json(output_dir / "environment.json", environment)

    best_accuracy = -1.0
    best_ckpt = output_dir / "resnet18_best.ckpt"
    metrics_path = output_dir / "metrics.jsonl"
    total_start = time.perf_counter()
    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        for epoch in range(1, args.epochs + 1):
            epoch_start = time.perf_counter()
            model.train(1, train_ds, dataset_sink_mode=False)
            evaluation = model.eval(eval_ds, dataset_sink_mode=False)
            record = {
                "epoch": epoch,
                "validation_accuracy": float(evaluation["accuracy"]),
                "epoch_seconds": round(time.perf_counter() - epoch_start, 4),
            }
            if record["validation_accuracy"] >= best_accuracy:
                best_accuracy = record["validation_accuracy"]
                save_checkpoint(network, str(best_ckpt))
                record["best_checkpoint_updated"] = True
            metrics_file.write(json.dumps(record) + "\n")
            metrics_file.flush()
            print(json.dumps(record), flush=True)

    summary = {
        "best_validation_accuracy": best_accuracy,
        "total_seconds": round(time.perf_counter() - total_start, 4),
        "best_checkpoint": str(best_ckpt),
        "best_checkpoint_sha256": sha256(best_ckpt),
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
