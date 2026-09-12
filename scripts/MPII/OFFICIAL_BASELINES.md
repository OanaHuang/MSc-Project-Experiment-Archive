# MPII official baseline reproduction

This series keeps the original random MPII split untouched and adds the fixed
`train.json` / `valid.json` protocol used by the official HRNet repository.

## Prepare the protocol

The official JSON annotations and the existing project metadata are local data
assets and are not committed. Prepare ordered JSONL files with:

```bash
python scripts/MPII/prepare_official_protocol.py
```

The expected validation population is 2,958 people. The generated manifest
records any unavailable official training image instead of silently changing
the protocol.

## Experiments

| ID | Model | Official reference |
|---|---|---|
| HB0 | Pose-ResNet-50 | MPII 256x256 checkpoint |
| HB1 | HRNet-W32 | MPII 256x256 checkpoint |
| HB2 | HRNet-W48 | MPII 256x256 checkpoint |

HB0 and HB1 use the original Microsoft/HRNet releases. The original HRNet
README lists an MPII W48 file, but that file is absent from the currently shared
model-zoo folder. HB2 therefore uses the official OpenMMLab MMPose HRNet-W48
MPII release (210 epochs, reported PCKh 90.1%) and records that distinct
provenance in its experiment YAML.

Evaluate a downloaded official checkpoint with:

```bash
python scripts/MPII/evaluate_official_baseline.py --experiment hb0
python scripts/MPII/evaluate_official_baseline.py --experiment hb1
python scripts/MPII/evaluate_official_baseline.py --experiment hb2
```

All models use the same official affine crop, quarter-pixel decoder, flip test,
heatmap shift, PCKh and project PCKHN implementation. HB0 preserves the original
SimpleBaseline BGR input convention; HB1/HB2 preserve HRNet's `COLOR_RGB=true`.
HB0's released checkpoint was trained for 140 epochs with learning-rate drops at
90 and 120. HB1/HB2 use the later HRNet schedule of 210 epochs with drops at 170
and 200. These values document checkpoint provenance; evaluation does not train
the model again. HB0 also verifies a canonical SHA-256 over its tensor names and
values, so equivalent checkpoints pass even when PyTorch serialization differs.
Results are written below
`Outputs_New/mpii/external_baselines/mpii_hrbase_v1`.

The adapted model code derives from the MIT-licensed
HRNet/HRNet-Human-Pose-Estimation repository. Preserve the repository copyright
and MIT license when redistributing these files.
