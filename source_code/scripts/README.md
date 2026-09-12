# SpikePose experiment framework

`scripts` separates the shared model from dataset-specific data, metrics and
visualization code. It is self-contained and does not import the legacy
`Scripts` package. Existing scripts and outputs are not modified.

## Layout

- `spikepose/models`: neurons, layers, blocks, backbones, necks and heads.
- `experiments`: shared canonical model configurations, including the
  controlled B0-to-E9 main research route.
- `ablations`: shared comparison groups that contain no dataset-specific
  temporal semantics.
- `MPII`: MPII data, evaluation, training, report visualization, and local
  `experiments`/`ablations` for the static-image T series.
- `NTU_RGBD`: NTU RGB+D data, evaluation, training, report visualization, and
  local `experiments`/`ablations` for real-frame temporal studies.
- `tools`: multi-GPU scheduling and result summaries.

## Train one model

MPII baseline:

```bash
python3 scripts/MPII/train.py --experiment baseline
```

NTU RGB+D baseline:

```bash
python3 scripts/NTU_RGBD/train.py --experiment baseline
```

`--epochs` is the target total epoch count. To continue a run that completed
20 epochs through epoch 40, reuse the same dataset, category, batch name,
experiment and seed, then add `--resume --epochs 40`. Training restores
`checkpoints/last.pt` and runs epochs 21 through 40. Omitting `--resume` starts
a newly initialized training run; use a new batch name when retaining the old
run is required.

## Run an ablation batch

```bash
python3 scripts/tools/run_ablation.py \
  --dataset mpii \
  --group structure \
  --batch-name mpii_140ep \
  --seeds 42 43 44 \
  --devices 0 1 2 3
```

Each GPU runs one independent experiment. The batch creates one shared
`sample_manifest.json`, so all runs visualize the same random ten samples.

## Outputs

Results use canonical path names while experiment/config IDs remain stable:

```text
Outputs_New/{dataset}/{category}/{batch}/{experiment}/seed_{seed}/
```

The ordered MPII B0-to-E9 route is stored together under:

```text
Outputs_New/mpii/architecture_evolution/profile_b/
├── b0_original_baseline/
├── e0_unified_baseline/
├── e1_single_scale_head/
├── e2_stage123_fusion/
├── e3_four_stage_fusion/
├── e4_membrane_residual/
├── e5_spike_fpn_ann_head/
├── e6_spike_fpn_linear_head/
├── e7_resformer_stage3/
├── e8_resformer_stage4/
└── e9_resformer_stages34/
```

Legacy categories, batch names and experiment IDs are accepted by the path
resolver, so existing commands resume these canonical runs instead of creating
a second legacy directory tree. Shared Pose and Backbone results are recorded
in their comparison directories through `references.json`, without duplicating
checkpoints.

Each run contains the resolved configuration, `best.pt`, `last.pt`, training
history, metrics and report images. Visualization runs once after training and
uses the best checkpoint. It renders the first ten validation samples and ten
fixed random validation samples. The labelled validation split is used because
the official MPII test annotations are not public.

Every run records layer-wise MACs, effective spike operations, firing rates,
45 nm analytical energy and the same-architecture ANN energy ratio. The report
uses 4.6 pJ per FP32 MAC and 0.9 pJ per spike-driven accumulation, following
the SpikeYOLO convention. Whole-GPU power
is sampled during training and best-model inference, then integrated into
joules and watt-hours under `analysis/hardware`. Hardware energy should be
measured on an otherwise idle GPU for a model-specific comparison.

## Stable inference energy remeasurement

The original post-training benchmark is retained for compatibility. Reliable
energy comparisons use the independent `measurement_v2` command, which does
not import or modify the legacy `Scripts` package. It records a 60-second idle
baseline, performs 100 warm-up iterations, measures five independent runs of
at least 60 seconds, reports gross and idle-subtracted energy, and flags an
energy coefficient of variation above 5% for review.

Do not run this command while T1, T4, or another GPU workload is active:

```bash
CUDA_VISIBLE_DEVICES=0 python3 scripts/tools/remeasure_inference.py \
  --run-dir Outputs_New/mpii/architecture_evolution/profile_b/e9_resformer_stages34/seed_42 \
  --physical-gpu 0 \
  --device cuda:0
```

Results are written under
`analysis/hardware/inference_remeasure/measurement_v2`. Existing inference
JSON and CSV files are never overwritten. Protocol defaults live in
`scripts/configs/measurement.yaml`.

Audit training-energy metadata without changing any run:

```bash
python3 scripts/tools/audit_energy_results.py \
  --output-root Outputs_New \
  --report Outputs_New/training_energy_audit.csv
```

New training runs record the requested, starting, completed, and measured
epoch counts. Energy captured only after resuming is explicitly marked
`partial` rather than being presented as complete training energy.

## Heatmap coordinate-decoding ablation

Coordinate decoding is an independent post-training ablation. It reads a
frozen source run but writes no files into that run. A local smoke test and a
formal server run use the same relative layout under `Outputs_New`:

```text
Outputs_New/mpii/heatmap_decoding_ablation/{v1_smoke|v1}/b0/seed_42/
```

Run a small local validation against a locally mirrored B0 run:

```bash
python3 scripts/MPII/heatmap_ablation.py \
  --source-run Server_outputs_New/mpii/architecture_evolution/profile_b/b0_original_baseline/seed_42 \
  --model-id b0 \
  --batch-name heatmap_v1_smoke \
  --max-validation-samples 32 \
  --output-root Outputs_New
```

Run the formal experiment on the server by changing only the source run and
batch name, and by omitting the sample limit:

```bash
python3 scripts/MPII/heatmap_ablation.py \
  --source-run Outputs_New/mpii/architecture_evolution/profile_b/b0_original_baseline/seed_42 \
  --model-id b0 \
  --batch-name heatmap_v1 \
  --device cuda:0 \
  --output-root Outputs_New
```

The first pass caches original-input and flipped-input heatmaps. Argmax,
Quarter-Pixel, DARK, UDP and Flip Shift variants then reuse that cache without
retraining or repeating model inference. Multi-model scheduling is available
through `scripts/tools/run_heatmap_ablation.py`; completed model runs can
be combined with `scripts/tools/summarize_heatmap_ablation.py`.

## Add an ablation

Create shared model YAML under `experiments`. Dataset-specific temporal YAML
belongs under `MPII/experiments` or `NTU_RGBD/experiments`; use
`dataset_scope: mpii` or `dataset_scope: ntu_rgbd` so cross-dataset commands
fail before training. Its `name` is used in code
outputs, visualizations and paper tables. A child configuration only states its
parent and changed values:

```yaml
id: example
name: SpikePose-Example
category: structure
parent: baseline
changes:
  model.backbone.output_stages: [1, 2, 3]
```

Add the canonical id to the matching shared or dataset-local `ablations`
directory to include it in a paper
comparison. References are reused without duplicate model configurations or
training. New components are registered in `spikepose/models/registry.py`.

The paper's ordered B0-to-E9 comparison is defined by `ablations/main_route.yaml`.
The existing legacy experiment ids remain available so completed runs and their
resolved configurations can still be reproduced.

`baseline` is the historical B0 trained with `pose_ablation`. The new `e0`, `e1`,
and `e2` configurations use the same `residual_bf` Profile B and `small_normal`
output initialization as the completed E3-to-E9 subchain. This makes E0-to-E9
a controlled architecture-evolution route after the B0-to-E0 training-recipe
intervention.

Generate and record two new, non-repeating training seeds with:

```bash
python3 scripts/tools/run_ablation.py \
  --dataset mpii \
  --group main_route \
  --batch-name mpii_main_route_20ep \
  --random-seeds 2 \
  --devices 0 1 2 3 \
  --epochs 120
```

The selected values are printed before training and saved under
`architecture_evolution/profile_b/seed_plans/`. Random seed mode refuses to
reuse any existing `seed_*` value and cannot be combined with `--resume`.

Before scheduling training, run the configuration-difference and compatibility
checks followed by the model forward-pass smoke test:

```bash
python3 scripts/experiment_integrity_test.py
python3 scripts/smoke_test.py
```

## Strict Profile B stage-fusion ablation

The S0-to-S6 study keeps a four-stage backbone, four projection branches,
additive fusion, head, parameter count and Profile B recipe fixed. A binary
mask alone controls whether each projected stage contributes to the sum. S0-S3
progressively add stages from the deepest feature; S4-S6 are leave-one-stage-out
variants. All seven formal runs use seed 42. The dedicated launcher first
trains every run through epoch 20. It then applies the same continuation gate
as the E series: every run must report `completed` and contain both `last.pt`
and `best.pt`. Only then does it resume the same optimizer, scheduler, RNG and
history state from epoch 21 through epoch 120:

```bash
python3 scripts/tools/run_stage_fusion.py --devices 0 1 2 3
```

Results are written to
`Outputs_New/mpii/stage_fusion/profile_b/` using the same semantic directory
convention as the E series:

```text
stage_fusion/profile_b/
├── sample_manifest.json
├── s0_stage4_only/seed_42/
├── s1_stages34/seed_42/
├── s2_stages234/seed_42/
├── s3_stages1234/seed_42/
├── s4_without_stage2/seed_42/
├── s5_without_stage3/seed_42/
└── s6_without_stage4/seed_42/
```

Each run uses the standard E-series artifact layout: resolved configuration,
`best.pt` and `last.pt`, training CSV, metrics, energy analysis, status and
visualizations.
The historical `stage1`, `stage12` and `stage123` runs remain unchanged Profile
A experiments and are not reused in this controlled comparison.

## MPII coordinate-representation series

The coordinate-classification (CC1-CC3) and direct-regression (RG1-RG2)
series are peer studies to the E series. Both inherit the accuracy-leading E9
model (`resformer_fpn`) and change only the prediction head and corresponding
supervision. The existing E9 heatmap result is the shared CC0/RG0 baseline and
is not retrained. All preliminary runs use Profile B for 20 epochs with seed 42:

```bash
python3 scripts/tools/run_coordinate_representation.py --dry-run
python3 scripts/tools/run_coordinate_representation.py
```

Outputs are separated by series under
`Outputs_New/mpii/{coordinate_classification|coordinate_regression}/profile_b_ep020_seed42/`.
