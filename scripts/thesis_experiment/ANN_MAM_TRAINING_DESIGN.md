# SpikePose-ANN-MAM training design

## Question and controlled comparison

This Pilot asks whether Core MAM improves temporal pose estimation when the
spatial SpikePose network uses ReLU activations rather than spiking neurons.
The primary comparison is seed-matched:

- control: `pilot20_ntu_spikepose_ann` (frame reset, no temporal memory);
- treatment: `pilot20_ntu_spikepose_ann_mam` (the same NTU-adapted spatial
  checkpoint plus Core MAM).

Both runs use the same NTU cross-subject metadata, MPII-16 joint mapping,
16-frame clips, augmentations, heatmap objective, validation split and seed 42.
Consequently, the treatment does not confound the MAM effect with a different
MPII initialization or a separately trained NTU spatial model.

This is a reproducible **MPII-16-layout Pilot**, matching the repository's
current MAM experiments. It is not a final NTU-18 mainline run: the NTU-18
joint list, flip pairs and normalization policy must be frozen first, as noted
in `NTU18_MAINLINE.md`.

## Dependency chain

1. Train `mpii_official_spikepose_ann` from scratch for 140 epochs in
   `Outputs_Thesis`.
2. Transfer its spatial weights to `pilot20_ntu_spikepose_ann` and adapt on NTU
   for 20 epochs in `Outputs_Thesis_Pilot20`.
3. Load the seed-matched NTU ANN checkpoint with `spatial_weights_only`, create
   a fresh zero-initialized MAM, and train
   `pilot20_ntu_spikepose_ann_mam` for 20 epochs.

At MAM initialization, residual gamma and the residual offset are zero, so the
treatment exactly reproduces the spatial ANN heatmaps before training.

## MAM optimization schedule

| Epochs | Trainable modules | Purpose |
|---|---|---|
| 1-5 | offset head, joint embedding | Learn alignment from the offset loss while preserving ANN output |
| 6-10 | previous modules, gate head, gamma | Introduce learned memory fusion |
| 11-20 | MAM, neck, head, final backbone stage | Joint low-rate adaptation without destabilizing early spatial features |

BatchNorm remains frozen in every MAM phase.  MAM uses full 16-frame BPTT,
dynamic per-joint decay, a 2-pixel residual-offset bound, and a `0.01` offset
loss. KPA, TPA and predictive losses are disabled so this run measures Core MAM
only.

## Launch and inspect

From `scripts/thesis_experiment`:

```bash
PYTHONPATH=src python -m spikepose_thesis plan \
  --study spikepose_ann_mam_pilot20

PYTHONPATH=src python tools/run_icassp2027_queue.py \
  --phase spikepose-ann-mam-pilot20 --gpus 0 --dry-run

PYTHONPATH=src python tools/run_icassp2027_queue.py \
  --phase spikepose-ann-mam-pilot20 --gpus 0
```

The queue respects all three checkpoint dependencies, skips completed runs and
resumes interrupted runs. Use more GPU IDs if available; dependency waves still
prevent the treatment from starting before its exact source is complete.

## Decision rule

Select checkpoints using validation PCKh only. After the Pilot checkpoint is
fixed, compare full-video test PCKh, velocity error, acceleration error and
theoretical cost against the ANN-frame control. Do not use the test set to tune
MAM. Promote to a multi-seed/formal run only if the validation gain is positive
and the full-video temporal metrics do not regress materially. Any promotion
must use a formal ANN-frame source trained under the same protocol; a Pilot
checkpoint must not initialize a formal run.

Because epochs 11-20 partially fine-tune the spatial model, the final comparison
measures the complete ANN+MAM training recipe rather than a frozen-backbone
MAM-only causal effect. If strict module attribution is required after this
screening run, add a frozen-spatial MAM ablation initialized from the same ANN
checkpoint; do not choose whether to run that ablation from test-set results.
