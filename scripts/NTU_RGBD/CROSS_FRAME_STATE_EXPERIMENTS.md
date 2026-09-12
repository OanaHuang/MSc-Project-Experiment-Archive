# NTU Cross-Frame State T-Series

This series evaluates DSTA-inspired causal joint-wise temporal models. All runs
use E6 (`mem_fpn`) and seed 42. T2-T5 use only the current and three preceding
frames; unlike the original DSTA, they never consume future frames.

Every run uses the same target-frame population: a current frame is
eligible only when the previous three contiguous source frames also exist.
Only the current frame supplies the heatmap target.

| ID | Input/control | Main comparison |
|---|---|---|
| T0 | current frame | spatial baseline |
| T1 | current repeated twice | repeated-compute temporal control |
| T2 | four causal frames | learned temporal weights for each joint |
| T3 | T2 + NTU skeleton graph | decoupled spatial contribution |
| T4 | four causal frames | confidence-gated joint-wise aggregation |
| T5 | four causal frames | motion-adaptive joint-wise aggregation |

List the machine-readable manifest:

```bash
.venv/bin/python scripts/NTU_RGBD/train_cross_frame_series.py --list
```

Dry-run or train one 20-epoch screening experiment:

```bash
.venv/bin/python scripts/NTU_RGBD/train_cross_frame_series.py \
  --experiment t2 --device cuda:0 --physical-gpu 0 --dry-run
```

Use `--epochs 120` only after the screening decision. The default source is
the deterministic S010 `clip4_256` set: four consecutive frames every 16
source frames, one shared person box per clip, and 256x256 cached crops.

```text
Outputs_New/ntu_rgbd/t_series/clip4_256_20ep/
  t0/seed_42/
  t1/seed_42/
  ...
  t5/seed_42/
```

Prepare the dataset once:

```bash
.venv/bin/python scripts/NTU_RGBD/prepare_clip4_256.py \
  --metadata Datasets/NTU_RGBD/metadata/s010/train_split.csv \
  --metadata Datasets/NTU_RGBD/metadata/s010/val_split.csv \
  --metadata Datasets/NTU_RGBD/metadata/s010/test_split.csv \
  --workers 8
```
