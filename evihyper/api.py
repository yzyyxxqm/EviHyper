import torch
from torch import Tensor

from ._model import Model
from .config import EviHyperConfig


class EviHyper(Model):
    """EviHyper with input validation and a target-free prediction interface."""

    def __init__(self, config: EviHyperConfig):
        super().__init__(config)

    def forward(self, x: Tensor, x_mask: Tensor | None = None, **kwargs):
        if x.ndim != 3 or min(x.shape) < 1 or x.shape[-1] != self.enc_in:
            raise ValueError("x must have shape [batch, history, enc_in]")
        if not x.is_floating_point():
            raise ValueError("x must be floating point")
        if x_mask is None:
            x_mask = torch.ones_like(x)
        if x_mask.shape != x.shape or x_mask.device != x.device:
            raise ValueError("x_mask must match the shape and device of x")
        if not torch.all((x_mask == 0) | (x_mask == 1)):
            raise ValueError("x_mask must contain only zero and one")
        if not torch.all(x_mask.flatten(1).any(dim=1)):
            raise ValueError("Each history must contain at least one observed measurement")
        if not torch.isfinite(x[x_mask.bool()]).all():
            raise ValueError("Observed measurements must be finite")
        if (kwargs.get("x_mark") is None) != (kwargs.get("y_mark") is None):
            raise ValueError("Provide x_mark and y_mark together")
        for key, length in (("x_mark", x.shape[1]), ("y_mark", self.pred_len)):
            marks = kwargs.get(key)
            if marks is not None:
                if marks.ndim != 3 or marks.shape[:2] != (x.shape[0], length) or marks.shape[2] < 1:
                    raise ValueError(f"{key} has an invalid batch, time, or feature dimension")
                if marks.device != x.device or not torch.isfinite(marks).all():
                    raise ValueError(f"{key} must be finite and on the same device as x")
        y = kwargs.get("y")
        if y is None:
            raise ValueError("forward requires y; use predict for target-free inference")
        if (
            y.ndim != 3
            or y.shape[:2] != (x.shape[0], self.pred_len)
            or y.shape[-1] not in {self.enc_in, self.c_out}
        ):
            raise ValueError("y must match the configured horizon and output channels")
        if y.device != x.device or not y.is_floating_point() or not torch.isfinite(y).all():
            raise ValueError("y must be finite floating-point values on the same device as x")
        y_mask = kwargs.get("y_mask")
        if y_mask is not None:
            if y_mask.shape != y.shape or y_mask.device != y.device:
                raise ValueError("y_mask must match the shape and device of y")
            if not torch.all((y_mask == 0) | (y_mask == 1)):
                raise ValueError("y_mask must contain only zero and one")
        sample_id = kwargs.get("sample_ID")
        station_enabled = self.configs.station_value_bias or self.configs.station_phase_bias
        if station_enabled and sample_id is None:
            raise ValueError("Station-dependent settings require sample_ID")
        if sample_id is not None:
            if (
                sample_id.shape != (x.shape[0],)
                or sample_id.dtype not in {torch.int32, torch.int64}
                or sample_id.device != x.device
                or (sample_id < 0).any()
            ):
                raise ValueError(
                    "sample_ID must contain one nonnegative integer per window on the input device"
                )
            if (
                station_enabled
                and (sample_id // self.station_time_steps >= self.station_count).any()
            ):
                raise ValueError("sample_ID exceeds the configured station range")
        return super().forward(x=x, x_mask=x_mask, **kwargs)

    @torch.no_grad()
    def predict(
        self,
        x: Tensor,
        x_mask: Tensor | None = None,
        x_mark: Tensor | None = None,
        y_mark: Tensor | None = None,
        sample_ID: Tensor | None = None,
    ) -> Tensor:
        was_training = self.training
        self.eval()
        try:
            return self(
                x=x,
                x_mask=x_mask,
                x_mark=x_mark,
                y_mark=y_mark,
                sample_ID=sample_ID,
                y=x.new_zeros(x.shape[0], self.pred_len, self.c_out),
                exp_stage="test",
            )["pred"]
        finally:
            self.train(was_training)
