import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from .train import file_digest


def read_source(path, dataset):
    if dataset == "etth1":
        with Path(path).open(newline="") as stream:
            reader = csv.DictReader(stream)
            if not reader.fieldnames:
                raise ValueError("ETTh1 CSV must contain a header and hourly records")
            variables = reader.fieldnames[1:]
            if reader.fieldnames[0] != "date" or len(variables) != 7:
                raise ValueError("ETTh1 CSV requires date followed by seven variables")
            rows = list(reader)
        values = np.array([[float(row[key]) for key in variables] for row in rows])[
            None
        ]
        masks = np.ones_like(values, dtype=np.float32)
        timestamps = np.array([row["date"] for row in rows])
        stations = ["ETTh1"]
        if len(rows) < 14400:
            raise ValueError("ETTh1 requires at least 14400 hourly rows")
    else:
        with np.load(path, allow_pickle=False) as archive:
            values = archive["values"].astype(np.float32)
            masks = archive["masks"].astype(np.float32)
            timestamps = archive["timestamps"]
            variables = archive["variables"].astype(str).tolist()
            stations = archive["station_ids"].astype(str).tolist()
    if values.ndim != 3 or min(values.shape) < 1 or masks.shape != values.shape:
        raise ValueError("values and masks must have shape [stations, time, channels]")
    if (
        not np.isin(masks, [0, 1]).all()
        or not np.isfinite(values[masks.astype(bool)]).all()
    ):
        raise ValueError("Masks must be binary and observed values must be finite")
    if len(variables) != values.shape[2] or len(stations) != values.shape[0]:
        raise ValueError("Variable and station names must match the value dimensions")
    if len(set(stations)) != len(stations) or len(set(variables)) != len(variables):
        raise ValueError("Station and variable names must be unique")
    times = np.char.replace(np.asarray(timestamps).astype(str), "Z", "").astype(
        "datetime64[s]"
    )
    if times.shape != (values.shape[1],) or np.isnat(times).any():
        raise ValueError("timestamps must contain one valid UTC time per row")
    if not np.all(np.diff(times) == np.timedelta64(1, "h")):
        raise ValueError(
            "These adapters require an hourly grid; insert masked rows for absent hours"
        )
    return values, masks, times, variables, stations


def calendar_marks(times, dataset):
    day = times.astype("datetime64[D]")
    year = times.astype("datetime64[Y]")
    day_phase = (times - day) / np.timedelta64(1, "D")
    days = (day - np.datetime64("1970-01-01")) / np.timedelta64(1, "D")
    if dataset == "etth1":
        marks = [
            day_phase * 24 / 23 - 0.5,
            (days + 3) % 7 / 6 - 0.5,
            (day - times.astype("datetime64[M]")) / np.timedelta64(1, "D") / 30 - 0.5,
            (day - year) / np.timedelta64(1, "D") / 365 - 0.5,
        ]
    else:
        year_phase = (times - year) / ((year + 1).astype("datetime64[s]") - year)
        marks = [year_phase, day_phase, (days % 7 + day_phase) / 7]
    return np.stack(marks, axis=-1).astype(np.float32)


def prepare_windows(
    path,
    output,
    dataset,
    seq_len=48,
    pred_len=24,
    missing_rate=0.0,
    train_fraction=1.0,
    data_seed=23,
):
    if seq_len < 1 or pred_len < 1 or not 0 <= missing_rate <= 0.95:
        raise ValueError(
            "Lengths must be positive and missing_rate must lie in [0, 0.95]"
        )
    if not 0 < train_fraction <= 1 or not 0 <= data_seed < 2**32:
        raise ValueError(
            "train_fraction must lie in (0, 1] and data_seed in [0, 2**32)"
        )
    values, masks, times, variables, stations = read_source(path, dataset)
    n_stations, n_steps, channels = values.shape
    if dataset == "etth1":
        train_end, val_end, test_end = 8640, 11520, 14400
    else:
        train_end = int(n_steps * 0.7)
        val_end, test_end = train_end + int(n_steps * 0.1), n_steps
    if (
        train_end < seq_len + pred_len
        or min(val_end - train_end, test_end - val_end) < pred_len
    ):
        raise ValueError(
            "Every split must be long enough for the requested forecast horizon"
        )
    mean = np.zeros(channels, dtype=values.dtype)
    std = np.ones(channels, dtype=values.dtype)
    for channel in range(channels):
        observed = values[:, :train_end, channel][
            masks[:, :train_end, channel].astype(bool)
        ]
        if observed.size:
            mean[channel] = observed.mean()
            spread = observed.std()
            std[channel] = spread if spread > 1e-6 else 1.0
    scaled = np.where(masks > 0, (values - mean) / std, 0).astype(np.float32)
    marks = calendar_marks(times, dataset)
    total = seq_len + pred_len
    relative = np.arange(total, dtype=np.float32)[:, None] / max(total - 1, 1)
    splits = {
        "train": (0, train_end),
        "val": (train_end - seq_len, val_end),
        "test": (val_end - seq_len, test_end),
    }
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    sizes = {}
    for split, (start, end) in splits.items():
        stride = 1 if dataset == "etth1" or split == "train" else 6
        windows = []
        for station in range(n_stations):
            for begin in range(start, end - total + 1, stride):
                split_at = begin + seq_len
                if masks[station, begin:split_at].sum() < max(
                    1, 0.05 * seq_len * channels
                ):
                    continue
                if masks[station, split_at : begin + total].sum() < max(
                    1, 0.05 * pred_len * channels
                ):
                    continue
                windows.append((station, begin))
        if split == "train" and train_fraction < 1 and windows:
            keep = np.random.default_rng(data_seed).choice(
                len(windows),
                max(1, int(np.ceil(len(windows) * train_fraction))),
                replace=False,
            )
            windows = [windows[index] for index in sorted(keep)]
        if not windows:
            raise ValueError(f"No eligible windows in {split}")
        arrays = {
            key: []
            for key in ("x", "y", "x_mask", "y_mask", "x_mark", "y_mark", "sample_ID")
        }
        for station, begin in windows:
            split_at, stop = begin + seq_len, begin + total
            sample_id = station * n_steps + begin
            x_mask = masks[station, begin:split_at].copy()
            if missing_rate:
                base = {"etth1": 1_000_003, "noaa": 3_000_003, "uscrn": 4_000_003}[
                    dataset
                ]
                generator = torch.Generator().manual_seed(
                    base + data_seed * 10_007 + sample_id
                )
                keep = (
                    torch.rand(x_mask.shape, generator=generator) < 1 - missing_rate
                ).numpy()
                keep[0] = True
                x_mask *= keep
            if not x_mask.any():
                raise ValueError(
                    f"Sparsification left an empty history at sample_ID={sample_id}"
                )
            local_marks = marks[begin:stop]
            if dataset != "etth1":
                local_marks = np.concatenate([relative, local_marks], axis=-1)
            sample = {
                "x": np.where(x_mask > 0, scaled[station, begin:split_at], 0),
                "x_mask": x_mask,
                "y": scaled[station, split_at:stop],
                "y_mask": masks[station, split_at:stop],
                "x_mark": local_marks[:seq_len],
                "y_mark": local_marks[seq_len:],
                "sample_ID": np.int64(sample_id),
            }
            for key, value in sample.items():
                arrays[key].append(value)
        np.savez_compressed(
            output / f"{split}.npz",
            **{key: np.stack(value) for key, value in arrays.items()},
        )
        sizes[split] = len(windows)
    metadata = {
        "dataset": dataset,
        "source_sha256": file_digest(path),
        "variables": variables,
        "station_ids": stations,
        "station_time_steps": n_steps,
        "seq_len": seq_len,
        "pred_len": pred_len,
        "missing_rate": missing_rate,
        "train_fraction": train_fraction,
        "data_seed": data_seed,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "split_indices": splits,
        "windows": sizes,
        "time_mark_mode": "prepend_relative" if dataset == "etth1" else "input_first",
        "split_sha256": {
            split: file_digest(output / f"{split}.npz") for split in splits
        },
    }
    (output / "preparation.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def main():
    parser = argparse.ArgumentParser(
        description="Prepare hourly EviHyper window splits"
    )
    parser.add_argument("--dataset", choices=("etth1", "noaa", "uscrn"), required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seq-len", type=int, default=48)
    parser.add_argument("--pred-len", type=int, default=24)
    parser.add_argument("--missing-rate", type=float, default=0.0)
    parser.add_argument("--train-fraction", type=float, default=1.0)
    parser.add_argument("--data-seed", type=int, default=23)
    args = parser.parse_args()
    result = prepare_windows(
        args.input,
        args.output,
        args.dataset,
        args.seq_len,
        args.pred_len,
        args.missing_rate,
        args.train_fraction,
        args.data_seed,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
