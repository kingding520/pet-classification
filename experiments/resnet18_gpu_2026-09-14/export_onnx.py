"""Export the stage-three GPU ResNet-18 checkpoint to a checked ONNX model."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn
from torchvision import models


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    args = parse_args()
    checkpoint_path = Path(args.checkpoint).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if payload.get("architecture") != "resnet18" or payload.get("class_to_idx") != {"Cat": 0, "Dog": 1}:
        raise ValueError("Checkpoint is not the expected ResNet-18 Cat/Dog model.")
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(payload["state_dict"])
    model.eval()

    torch.manual_seed(1)
    fixed_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)
    with torch.inference_mode():
        torch_output = model(fixed_input).cpu().numpy()

    onnx_path = output_dir / "resnet18_pet.onnx"
    torch.onnx.export(
        model,
        (fixed_input,),
        onnx_path,
        input_names=["input"],
        output_names=["logits"],
        opset_version=17,
        dynamo=False,
    )
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_output = session.run(["logits"], {"input": fixed_input.numpy()})[0]
    max_abs_diff = float(np.max(np.abs(torch_output - onnx_output)))
    if not np.allclose(torch_output, onnx_output, rtol=1e-4, atol=1e-5):
        raise RuntimeError(f"ONNX output differs from PyTorch (max abs diff {max_abs_diff}).")

    np.save(output_dir / "fixed_input.npy", fixed_input.numpy())
    np.save(output_dir / "pytorch_logits.npy", torch_output)
    summary = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "onnx": str(onnx_path),
        "onnx_sha256": sha256(onnx_path),
        "opset": 17,
        "input_name": "input",
        "output_name": "logits",
        "input_shape": [1, 3, 224, 224],
        "class_to_idx": payload["class_to_idx"],
        "checkpoint_epoch": payload["epoch"],
        "checkpoint_validation_accuracy": payload["validation_accuracy"],
        "onnxruntime_max_abs_diff": max_abs_diff,
        "onnxruntime_providers": session.get_providers(),
    }
    (output_dir / "onnx_export_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
