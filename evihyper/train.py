import argparse
import hashlib
import json
import math
import platform
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from . import EviHyper, EviHyperConfig, ForecastLoss
from .data import WindowDataset


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def to_device(batch, device):
    return {key: value.to(device) for key, value in batch.items()}


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    squared, absolute, count = 0.0, 0.0, 0
    for batch in loader:
        batch = to_device(batch, device)
        pred = model.predict(
            batch["x"],
            batch["x_mask"],
            batch.get("x_mark"),
            batch.get("y_mark"),
            batch.get("sample_ID"),
        )
        if not torch.isfinite(pred).all():
            raise RuntimeError("Non-finite prediction")
        error = (pred - batch["y"])[batch["y_mask"].bool()].double()
        squared += error.square().sum().item()
        absolute += error.abs().sum().item()
        count += error.numel()
    if not count:
        raise ValueError("Evaluation requires observed targets")
    return {"mse": squared / count, "mae": absolute / count, "observed_targets": count}


def main():
    parser = argparse.ArgumentParser(description="Train the complete EviHyper model")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=7e-4)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--loss", choices=("mse", "mse_aux"), default="mse_aux")
    parser.add_argument(
        "--schedule", choices=("constant", "delayed"), default="delayed"
    )
    args = parser.parse_args()
    if (
        min(args.epochs, args.patience, args.batch_size, args.threads) < 1
        or args.workers < 0
    ):
        parser.error(
            "Epochs, patience, batch size and threads must be positive; workers cannot be negative"
        )
    if not math.isfinite(args.lr) or args.lr <= 0 or not 0 <= args.seed < 2**32:
        parser.error("lr must be positive and finite; seed must lie in [0, 2**32)")
    if args.train.resolve() == args.val.resolve():
        parser.error("Training and validation must use separate splits")
    hashes = {key: file_digest(getattr(args, key)) for key in ("train", "val")}
    if hashes["train"] == hashes["val"]:
        parser.error("Training and validation files are identical")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    config = EviHyperConfig(**json.loads(args.config.read_text()))
    training, validation = WindowDataset(args.train), WindowDataset(args.val)
    for dataset in (training, validation):
        dataset.check_config(config)
    args.output.mkdir(parents=True, exist_ok=False)
    run = {
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "model_config": config.to_dict(),
        "data_sha256": hashes,
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "numpy": np.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else str(device),
    }
    (args.output / "run.json").write_text(json.dumps(run, indent=2) + "\n")
    training = DataLoader(
        training,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        generator=torch.Generator().manual_seed(args.seed),
    )
    validation = DataLoader(
        validation, batch_size=args.batch_size, num_workers=args.workers
    )
    model, criterion = EviHyper(config).to(device), ForecastLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda epoch: (
            1.0 if args.schedule == "constant" or epoch < 2 else 0.8 ** (epoch - 2)
        ),
    )
    best, stale = float("inf"), 0
    with (args.output / "history.jsonl").open("w") as log:
        for epoch in range(1, args.epochs + 1):
            model.train()
            for batch in training:
                optimizer.zero_grad(set_to_none=True)
                result = model(**to_device(batch, device), exp_stage="train")
                # The mse setting excludes auxiliary supervision without changing the model.
                loss = criterion(
                    **result, exp_stage="train" if args.loss == "mse_aux" else "val"
                )["loss"]
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite training loss")
                loss.backward()
                if any(
                    p.grad is not None and not torch.isfinite(p.grad).all()
                    for p in model.parameters()
                ):
                    raise RuntimeError("Non-finite gradient")
                optimizer.step()
            metrics = evaluate(model, validation, device)
            record = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"], **metrics}
            line = json.dumps(record)
            print(line, flush=True)
            log.write(line + "\n")
            log.flush()
            if metrics["mse"] < best:
                best, stale = metrics["mse"], 0
                torch.save(
                    {
                        "config": config.to_dict(),
                        "model": model.state_dict(),
                        "epoch": epoch,
                        "validation": metrics,
                        "seed": args.seed,
                        "data_sha256": hashes,
                    },
                    args.output / "best.pt",
                )
            else:
                stale += 1
            if stale >= args.patience:
                break
            scheduler.step()


if __name__ == "__main__":
    main()
