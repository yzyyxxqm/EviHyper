from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class WindowDataset(Dataset):
    """Load split-specific windows that have already been scaled and masked."""

    def __init__(self, path: str | Path):
        required = {"x", "x_mask", "y", "y_mask"}
        allowed = required | {"x_mark", "y_mark", "sample_ID"}
        with np.load(path, allow_pickle=False) as archive:
            missing = required - set(archive.files)
            if missing:
                raise ValueError(f"Missing arrays: {sorted(missing)}")
            if "sample_ID" in archive:
                ids = archive["sample_ID"]
                if not np.issubdtype(ids.dtype, np.integer) or np.any(ids < 0):
                    raise ValueError("sample_ID must contain nonnegative integers")
            self.arrays = {
                key: torch.as_tensor(
                    np.array(archive[key], copy=True),
                    dtype=torch.long if key == "sample_ID" else torch.float32,
                )
                for key in allowed.intersection(archive.files)
            }
        x = self.arrays["x"]
        if x.ndim != 3 or min(x.shape) < 1:
            raise ValueError(
                "x must have nonempty [samples, history, channels] dimensions"
            )
        for key, array in self.arrays.items():
            if array.ndim == 0 or array.shape[0] != len(x):
                raise ValueError(f"{key} must have the same number of samples as x")
        for value_key, mask_key in (("x", "x_mask"), ("y", "y_mask")):
            values, mask = self.arrays[value_key], self.arrays[mask_key]
            if values.ndim != 3 or min(values.shape) < 1:
                raise ValueError(
                    f"{value_key} must be a nonempty three-dimensional array"
                )
            if mask.shape != values.shape or not torch.all((mask == 0) | (mask == 1)):
                raise ValueError(
                    f"{mask_key} must be a binary mask matching {value_key}"
                )
            if not torch.isfinite(values[mask.bool()]).all():
                raise ValueError(f"Observed {value_key} values must be finite")
            if not mask.flatten(1).any(dim=1).all():
                raise ValueError(
                    f"Every {value_key} window must contain an observed value"
                )
            self.arrays[value_key] = torch.where(
                mask.bool(), values, torch.zeros_like(values)
            )
        if ("x_mark" in self.arrays) != ("y_mark" in self.arrays):
            raise ValueError("Provide x_mark and y_mark together")
        for key, values in (("x_mark", x), ("y_mark", self.arrays["y"])):
            if key in self.arrays:
                marks = self.arrays[key]
                if (
                    marks.ndim != 3
                    or marks.shape[:2] != values.shape[:2]
                    or marks.shape[2] < 1
                    or not torch.isfinite(marks).all()
                ):
                    raise ValueError(
                        f"{key} must be finite and match its value time dimensions"
                    )
        if "sample_ID" in self.arrays and self.arrays["sample_ID"].ndim != 1:
            raise ValueError("sample_ID must contain one integer per window")

    def check_config(self, config):
        if self.arrays["x"].shape[1:] != (config.seq_len, config.enc_in):
            raise ValueError("Input dimensions do not match the model config")
        if self.arrays["y"].shape[1:] != (config.pred_len, config.c_out):
            raise ValueError("Target dimensions do not match the model config")
        if config.station_value_bias or config.station_phase_bias:
            ids = self.arrays.get("sample_ID")
            if (
                ids is None
                or (ids // config.station_time_steps >= config.station_count).any()
            ):
                raise ValueError(
                    "Station-dependent settings require valid sample_ID values"
                )

    def __len__(self):
        return self.arrays["x"].shape[0]

    def __getitem__(self, index):
        return {key: value[index] for key, value in self.arrays.items()}
