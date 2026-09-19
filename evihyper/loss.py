import torch
from torch import Tensor, nn


class ForecastLoss(nn.Module):
    """Masked forecast error with auxiliary supervision during training."""

    def forward(
        self,
        pred: Tensor,
        true: Tensor,
        mask: Tensor | None = None,
        aux_loss: Tensor | None = None,
        exp_stage: str = "train",
        **kwargs,
    ) -> dict[str, Tensor]:
        mask = torch.ones_like(true) if mask is None else mask
        squared_error = ((pred - true) * mask).square()
        mse = squared_error.sum() / mask.sum().clamp_min(1.0)
        loss = mse + aux_loss if exp_stage == "train" and aux_loss is not None else mse
        return {"loss": loss, "loss_mse": mse.detach()}
