# EviHyper

PyTorch implementation of EviHyper for forecasting irregular multichannel signals.

## Installation

Python 3.10+ and PyTorch 2.5+ are required.

```bash
python -m pip install -e '.[experiments]'
python examples/smoke.py
```

## Example: NOAA-Sp50

Dataset definitions and experimental protocols are described in the paper. Download, prepare and run the full model:

```bash
python examples/prepare_stations.py --dataset noaa --raw-dir data/raw/noaa \
  --output data/noaa.npz --download
python -m evihyper.prepare --dataset noaa --input data/noaa.npz \
  --output data/noaa_sp50 --missing-rate 0.5 --train-fraction 0.03 --data-seed 23
python -m evihyper.train --train data/noaa_sp50/train.npz --val data/noaa_sp50/val.npz \
  --config configs/noaa_sparse.json --output runs/noaa_sp50_seed2024 --seed 2024 --device cuda:0
python -m evihyper.evaluate --checkpoint runs/noaa_sp50_seed2024/best.pt \
  --data data/noaa_sp50/test.npz --output runs/noaa_sp50_seed2024/test.json --device cuda:0
```

Model settings are in [configs](configs/). ETTh1 uses `--lr 0.001 --loss mse`; other supplied profiles use the training defaults. Use `--device cpu` without a GPU and `--help` for command options. Output paths must be new.

Data and pretrained weights are not included. USHCN requires separately prepared NPZ splits. Baseline and ablation runners are not included.

The presets use the revised ETTh1 time adapter and a natural-NOAA station-value bias scale of 0.5 (1.0 in the original profile). This compact runner does not guarantee exact reproduction of the reported scores.
