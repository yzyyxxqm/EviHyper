import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from . import EviHyper, EviHyperConfig
from .data import WindowDataset
from .train import evaluate, file_digest


def main():
    parser = argparse.ArgumentParser(description="Evaluate a saved EviHyper checkpoint")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if min(args.batch_size, args.threads) < 1:
        parser.error("batch-size and threads must be positive")
    if args.output.exists():
        parser.error("Output already exists; choose a new path")
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = EviHyperConfig(**checkpoint["config"])
    dataset = WindowDataset(args.data)
    dataset.check_config(config)
    model = EviHyper(config).to(device)
    model.load_state_dict(checkpoint["model"])
    metrics = evaluate(model, DataLoader(dataset, batch_size=args.batch_size), device)
    report = {
        **metrics,
        "checkpoint_epoch": checkpoint["epoch"],
        "data_sha256": file_digest(args.data),
        "checkpoint_sha256": file_digest(args.checkpoint),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
