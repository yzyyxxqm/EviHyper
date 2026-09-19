import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .config import EviHyperConfig


def hamilton_product(lhs: Tensor, rhs: Tensor) -> Tensor:
    lr, li, lj, lk = lhs.chunk(4, dim=-1)
    rr, ri, rj, rk = rhs.chunk(4, dim=-1)
    return torch.cat(
        [
            lr * rr - li * ri - lj * rj - lk * rk,
            lr * ri + li * rr + lj * rk - lk * rj,
            lr * rj - li * rk + lj * rr + lk * ri,
            lr * rk + li * rj - lj * ri + lk * rr,
        ],
        dim=-1,
    )


def quaternion_conjugate(x: Tensor) -> Tensor:
    r, i, j, k = x.chunk(4, dim=-1)
    return torch.cat([r, -i, -j, -k], dim=-1)


class QuaternionLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        if in_features % 4 != 0 or out_features % 4 != 0:
            raise ValueError("QuaternionLinear requires dimensions divisible by 4.")
        self.r = nn.Linear(in_features // 4, out_features // 4)
        self.i = nn.Linear(in_features // 4, out_features // 4)
        self.j = nn.Linear(in_features // 4, out_features // 4)
        self.k = nn.Linear(in_features // 4, out_features // 4)

    def forward(self, x: Tensor) -> Tensor:
        r, i, j, k = x.chunk(4, dim=-1)
        return torch.cat(
            [
                self.r(r) - self.i(i) - self.j(j) - self.k(k),
                self.r(i) + self.i(r) + self.j(k) - self.k(j),
                self.r(j) - self.i(k) + self.j(r) + self.k(i),
                self.r(k) + self.i(j) - self.j(i) + self.k(r),
            ],
            dim=-1,
        )


class CausalAnchor(nn.Module):
    def __init__(self, configs: EviHyperConfig):
        super().__init__()
        self.projection_seq_len = configs.seq_len
        self.projection_pred_len = configs.pred_len
        self.projection = nn.Linear(self.projection_seq_len, self.projection_pred_len)

    def forward(self, x: Tensor, x_mask: Tensor, pred_len: int) -> Tensor:
        last = self._last_observed(x, x_mask).unsqueeze(1).expand(-1, pred_len, -1)
        filled = self._forward_fill(x, x_mask)
        projected = self._projection_anchor(filled, x_mask, pred_len)
        return torch.where(torch.isfinite(projected), projected, last)

    def _projection_anchor(self, filled: Tensor, x_mask: Tensor, pred_len: int) -> Tensor:
        if filled.shape[1] < self.projection_seq_len:
            pad = self.projection_seq_len - filled.shape[1]
            history = F.pad(filled.transpose(1, 2), (0, pad))
            mask_history = F.pad(x_mask.transpose(1, 2), (0, pad))
        else:
            history = filled[:, -self.projection_seq_len :, :].transpose(1, 2)
            mask_history = x_mask[:, -self.projection_seq_len :, :].transpose(1, 2)
        denom = mask_history.sum(dim=-1, keepdim=True).clamp_min(1.0)
        mean = (history * mask_history).sum(dim=-1, keepdim=True) / denom
        centered = (history - mean) * mask_history
        scale = torch.sqrt((centered.square().sum(dim=-1, keepdim=True) / denom).clamp_min(1e-4))
        projected = self.projection((history - mean) / scale).transpose(1, 2)
        projected = projected * scale.transpose(1, 2) + mean.transpose(1, 2)
        if projected.shape[1] >= pred_len:
            return projected[:, :pred_len, :]
        pad = pred_len - projected.shape[1]
        return F.pad(projected.transpose(1, 2), (0, pad)).transpose(1, 2)

    @staticmethod
    def _last_observed(x: Tensor, x_mask: Tensor) -> Tensor:
        last = torch.zeros(x.shape[0], x.shape[2], device=x.device, dtype=x.dtype)
        for idx in range(x.shape[1]):
            observed = x_mask[:, idx, :] > 0
            last = torch.where(observed, x[:, idx, :], last)
        return last

    @staticmethod
    def _forward_fill(x: Tensor, x_mask: Tensor) -> Tensor:
        filled = torch.zeros_like(x)
        last = torch.zeros(x.shape[0], x.shape[2], device=x.device, dtype=x.dtype)
        for idx in range(x.shape[1]):
            observed = x_mask[:, idx, :] > 0
            last = torch.where(observed, x[:, idx, :], last)
            filled[:, idx, :] = last
        return filled
