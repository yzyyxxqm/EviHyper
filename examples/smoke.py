import torch

from evihyper import EviHyper, EviHyperConfig, ForecastLoss


def main():
    torch.set_num_threads(2)
    torch.manual_seed(2024)
    config = EviHyperConfig(seq_len=12, pred_len=4, enc_in=3, d_model=16, d_ff=32, dropout=0)
    model = EviHyper(config).cpu()
    x = torch.randn(2, 12, 3)
    mask = (torch.rand_like(x) > 0.5).float()
    mask[:, 0] = 1
    y = torch.randn(2, 4, 3)
    result = model(x=x, x_mask=mask, y=y, exp_stage="train")
    loss = ForecastLoss()(**result, exp_stage="train")["loss"]
    loss.backward()
    prediction = model.predict(x, mask)
    assert torch.isfinite(loss) and torch.isfinite(prediction).all()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    print(f"CPU forward/backward passed; forecast shape: {tuple(prediction.shape)}")


if __name__ == "__main__":
    main()
