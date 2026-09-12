# SpikePose thesis experiments

This package is the paper-only training and evaluation workspace. Runtime code
under `src/spikepose_thesis` is standalone and must not import the historical
`Scripts` or `scripts` packages.

## Joint-layout policy

- **Main direction:** NTU cross-subject experiments will use a fixed NTU-18
  layout.
- **Retained auxiliary line:** the native NTU-25 Pilot8 preview remains in
  `ntu25_preview8` and must continue to use `ntu25_identity_v1`.
- **Migration safety:** the exact NTU-18 joint list, ordering, flip pairs and
  evaluation normalizer are not yet defined in this repository. The current
  ICASSP2027 NTU configs still resolve to the earlier MPII-compatible 16-joint
  target and are retained only as reproducible baselines. Do not launch them as
  final NTU-18 training jobs.

See [`NTU18_MAINLINE.md`](NTU18_MAINLINE.md) for the decisions required before
NTU-18 training can start.

## Canonical experiment entry points

- `configs/studies/icassp2027_pilot20.yaml`
- `configs/studies/icassp2027_confirm140.yaml`
- `configs/studies/icassp2027_pilot_eval.yaml`
- `configs/studies/icassp2027_confirm_eval.yaml`
- `configs/studies/ntu25_preview8.yaml`

The queue implementation is `tools/run_icassp2027_queue.py`; NTU-25 previews
use `tools/run_ntu25_preview_queue.py`. Historical P-series, transfer,
two-clock and learned-refinement schedules are intentionally not part of this
branch's experiment surface.

## Installation and validation

```bash
python -m pip install -e scripts/thesis_experiment
spikepose-thesis validate
spikepose-thesis preflight-data --workers 4
spikepose-thesis prepare-runtime-cache --setups full --workers 4
spikepose-thesis audit-data
```

Without installation:

```bash
PYTHONPATH=scripts/thesis_experiment/src \
python -m spikepose_thesis validate
```

The CLI defaults to the current paper studies:

```bash
spikepose-thesis plan --study icassp2027_pilot20
spikepose-thesis commands --study icassp2027_pilot20
spikepose-thesis report --study icassp2027_confirm140 \
  --split test --output Outputs_Thesis/tables
```

After a checkpoint is frozen, compute the activity-weighted theoretical energy
from real validation or test windows with:

```bash
spikepose-thesis profile \
  --experiment <experiment-id> --seed <seed> --split test \
  --device cuda:0 --batch-size 8
```

The command loads `checkpoints/best.pt` and writes
`analysis/theoretical_energy_<split>.json` inside the run directory. Use
`--max-samples` or `--max-batches` only for a documented pilot estimate; omit
both for the paper value. Reported energy follows the 45-nm FP32 convention of
4.6 pJ per MAC and 0.9 pJ per spike-triggered AC, averaged per real input
window. Memory access, data movement, nonlinear operations and neuron-state
updates are listed as excluded costs in the generated JSON.

For the frozen 16-joint paper baseline workflow and dependency waves, see
[`ICASSP2027_WORKFLOW.md`](ICASSP2027_WORKFLOW.md). Training artifacts belong in
`Outputs_Thesis` or `Outputs_Thesis_Pilot20`, never in this source directory.

## Data and evaluation invariants

- MPII uses 16 joints, 256×256 input and 64×64 heatmaps.
- NTU uses the official cross-subject split as the paper protocol.
- Every temporal clip uses one shared tube crop and synchronized augmentation.
- NTU predictions are restored to full-frame pixels before pose and temporal
  metrics are computed.
- Pilot and formal outputs are isolated; Pilot test evaluation is blocked.
- Validation selects checkpoints and filter parameters; test is evaluated once.
- The runtime cache stores the raw NTU-25 pose and can serve the future NTU-18
  subset and the retained native NTU-25 preview without rebuilding RGB crops.

## MAM complete-video reset-policy validation

Use `tools/validate_mam_reset_policy.py` to compare the frozen MAM checkpoint
under trained-window resets and video-boundary-only resets. The tool computes
one raw spatial heatmap stream using a single tube crop for each complete
video, then applies both reset policies to that same stream. Existing
fixed-context evaluations and their window-level crop geometry are not
modified.

Run a smoke test before the exhaustive evaluation:

```bash
PYTHONPATH=scripts/thesis_experiment/src \
python scripts/thesis_experiment/tools/validate_mam_reset_policy.py \
  --experiment mamv2_fullcs20 \
  --resolved-config <mam-run>/resolved_config.yaml \
  --seed 42 \
  --checkpoint <mam-run>/checkpoints/best.pt \
  --output Outputs_Thesis_Pilot20/reset_policy_validation/smoke \
  --device cuda:0 \
  --max-videos 20
```

Remove `--max-videos` and choose a new output directory for the exhaustive
4,563-video run. The output contains aligned `raw`, `window_reset`, and
`video_reset` prediction archives, per-method summaries, provenance, paired
video bootstrap intervals, and transition errors at the trained-window output
stitches. Compare reset policies only within this matched video-crop run;
their crop geometry intentionally differs from the earlier window-crop table.

## GT video-speed stratification

Use `tools/analyze_speed_strata.py` after full-video predictions are available.
The MTPose-inspired protocol computes consecutive-frame GT joint displacement,
normalizes it by the visible GT pose-box diagonal, averages it per video, and
forms deterministic equal-count low/medium/high tertiles. Predictions never
affect stratum assignment. Every compared archive must contain exactly the same
test rows and ground truth as the reference archive.

```bash
PYTHONPATH=scripts/thesis_experiment/src \
python scripts/thesis_experiment/tools/analyze_speed_strata.py \
  --reference <table4-root>/mam/raw/predictions_per_frame.npz \
  --prediction mam_window=<table4-root>/mam/window_reset/predictions_per_frame.npz \
  --prediction mam_video=<table4-root>/mam/video_reset/predictions_per_frame.npz \
  --prediction no_alignment_video=<table4-root>/no_alignment/video_reset/predictions_per_frame.npz \
  --output <table4-root>/speed_strata
```

Repeat `--prediction LABEL=PATH` for all Table-4 variants and filtering
baselines. The output contains `video_speed_strata.csv`,
`speed_strata_metrics.csv`, and a provenance-rich
`speed_strata_metrics.json`. Use `--normalization head_bone` only for a
documented sensitivity analysis; the default is the bbox-normalized movement
scale closest to the MTPose analysis.
