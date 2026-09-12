# NTU external human-pose baseline series

NHB compares the existing single-frame SpikePose FP0 reference with common ANN
pose architectures. All trainable baselines use the same S010 `clip4_256`
population, current frame, 256x256 input, 25 NTU heatmaps, seed 42, disabled
augmentation, FP0's `residual_bf` optimizer/scheduler profile, and PCKhn
evaluation as FP0.

| ID | Model | Initialization |
|---|---|---|
| NHB0 | SpikePose E6 / FP0 | scratch (existing reference) |
| NHB1 | Pose-ResNet-50 | scratch |
| NHB2 | Pose-ResNet-50 | official MPII HB0 |
| NHB3 | HRNet-W32 | scratch |
| NHB4 | HRNet-W32 | official MPII HB1 |
| NHB5 | HRNet-W48 | scratch |
| NHB6 | HRNet-W48 | official OpenMMLab MPII HB2 |

NHB0 is the existing FP0 result and is not duplicated by this launcher. List or
inspect commands with:

```bash
.venv/bin/python scripts/NTU_RGBD/train_nhb_series.py --list
.venv/bin/python scripts/NTU_RGBD/train_nhb_series.py \
  --experiment nhb1 --epochs 20 --device cuda:0 --dry-run
```

Run a one-epoch smoke test before the 20-epoch screening run:

```bash
.venv/bin/python scripts/NTU_RGBD/train_nhb_series.py \
  --experiment nhb1 --epochs 1 --max-train-samples 32 \
  --max-validation-samples 16 --device cuda:0
```

Outputs use
`Outputs_New/ntu_rgbd/external_baselines/clip4_256_<epochs>ep/<id>`.
Pretrained experiments require the corresponding official MPII checkpoint under
`Datasets/MPII/official_simplebaseline`. Every such run writes
`initialization.json`, including the checkpoint tensor fingerprint, every loaded
tensor, every shape mismatch, and every randomly initialized tensor. The
16-joint MPII final layer is deliberately skipped and the 25-joint NTU final
layer remains random.
