# MAM softmax fix: user-proposed coordinate decoding experiments

This category implements the methods in `SpikePose_MAM_Coordinate_Decoding_Fix.md`.
The reviewed source document SHA-256 is
`b4cad05ff58abc52ff92c069e0d1b8efb70e88c423d491afe900b0a196c517cd`.
It is a method repair experiment, not a new backbone or a heatmap motion network.

## Model identity and formulas

The spatial source is the **independently trained** `mamv2_fullcs_p00_source`,
seed 42, best checkpoint SHA-256
`004ca2da72c22d8c77ccf9688a3e325adbbd61eada16a8df0842318d6d2f4990`.
The scripts check this hash. Only spatial weights are loaded; MAM is initialized
fresh. This is the existing v2.6 MPII-style **16-joint** NTU-CS experiment, not
the planned NTU-18 migration or NTU-25 preview.

For `R=max(H,0)` the candidates are:

| Label | Weights before normalization | Role |
|---|---|---|
| softmax | exp(H) | matched retrained control |
| scaled10 / scaled20 | exp(beta H), beta=10 / 20 | temperature diagnostics |
| relu | R | direct nonnegative integral |
| power2 | R^2 | fixed power integral |
| expm1 | expm1(R), beta=1 | BCIR-inspired discrete background subtraction |
| dark | existing DARK implementation, kernel=11 | stopped-gradient coordinate reference |

All integral methods use `c=sum(P*x)`. Displacements remain heatmap pixels,
`d_coarse=c_t-c_(t-1)`, `d_final=d_coarse+2*tanh(offset_head(z))`.
Every **new training variant**, including the softmax control, uses
`q=max(ReLU(H))` as an uncalibrated response feature. No token normalization,
new confidence network, altered residual range, coordinate loss, or raw-heatmap
loss is added in this first study. Heatmaps used in storage/warp/fusion remain raw.
The historical model code path stays unchanged unless `mam_softmax_fix` is set
in the resolved experiment dictionary. Use the public `models.build_model` API
with the complete saved config to reconstruct this opt-in adapter.

## Numerical and invalid-input rules

Decoder calculations use FP32 outside autocast (FP64 is retained in tests).
Power weights are divided by the common peak before exponentiation; this leaves
normalized probabilities unchanged. Expm1 uses the exactly proportional stable
expression `exp(u-max(u)) * (-expm1(-u))`, **not** `expm1(u-max(u))`.
Positive-response mass must exceed `1e-8`; otherwise xy is a centre sentinel and
q=0. Such pairs have coarse displacement zero, history weight zero, and output/
memory equal to the current raw map. They remain in GT-valid aggregate errors;
invalid counts are reported. Nonfinite raw maps fail rather than entering fusion.
The raw peak itself is not a calibrated correctness probability.

## Phase 1: fixed validation heatmaps

Select two videos per action from the usable canonical validation dataset using
seed 42 before inference. Use the available two **independent contiguous 16-frame
clips** per video, fixed clip tube crop, no augmentation. No differences cross
clips. This is **not complete-video evaluation** and no test data are used.
All decoders share exactly the same FP32 spatial heatmaps. Store them once with
targets, clip identities and SHA-256 hashes (roughly 1 GiB for 120 videos).
Subsequent frozen MAM diagnostics reuse this cache. Training retains the existing
full-CS sampled-clip protocol and synchronized augmentation, with AMP enabled.

Report coordinate EPE, invalid fraction, coarse EPE, zero EPE, Coverage2,
unavoidable EPE outside the residual box, and predicted motion magnitude. Report
all pairs and GT displacement bins [0,.5), [.5,2), [2,4), [4,infinity) separately.
Synthetic checks reproduce the document table and combine amplitude changes
with background, secondary peaks and boundaries. DARK runs the existing full
preprocessing, so its synthetic result is not mislabeled as just the Taylor core.

Select two usable differentiable candidates among relu/power2/expm1 by the sum
of ascending ranks on coordinate EPE, equal-bin mean coarse EPE, equal-bin mean
reachable EPE, and invalid fraction; ties use candidate name. Zero motion is a
reference, never a hard admission gate. This is a declared screening heuristic,
not proof that the selected candidate will train best. DARK and the softmax
control always enter phase 2.

## Phase 2: frozen spatial training

Four jobs: matched softmax + DARK + two screened candidates, seed 42, 20 epochs.
Backbone, neck, head and BN buffers are frozen; evaluation verifies equality to
the source checkpoint. Epochs 1–5 train offset head and joint embedding with
gamma=0. Epochs 6–20 train offset, embedding, gate and gamma. Learning rate is
2e-4 for these modules with the inherited scheduler. Use the existing fused
heatmap MSE + 0.01 * final-offset SmoothL1, GT-valid masks and training protocol.
Full canonical validation selects best checkpoints; the fixed diagnostic subset
also reports PCKHB/MPJVE/MPJAccE and raw/final displacement metrics. These subset
scores are frame/pair weighted in original RGB pixels, not the paper full-video
macro scores. Each aligned checkpoint also gets a same-weight warp-disabled
diagnostic that must not be confused with a separately trained noalign model.

## Phase 3: alignment mechanism controls

Choose a candidate within 0.2 PCKHB percentage points of the retrained softmax
control on the fixed validation subset; among eligible candidates prefer lower
MPJAccE, then MPJVE, then higher PCKHB. If none qualifies, choose the highest-PCKHB
candidate for diagnosis and flag the failure to qualify (do not declare success).
Run that decoder with noalign and coarse-only controls, also 20 epochs. These
have no trainable offset head, so they train embedding/gate/gamma from epoch 1
and disable the offset loss. This necessary supervision/schedule difference is
explicit: matching a vacuous offset-only warmup would supply no learning signal.
Neither control removes residual **output fusion**. Add paired video bootstrap
intervals (2000 samples, seed 42); these are exploratory validation intervals,
not independent test evidence or training-seed variability estimates.

Joint fine-tuning and L_raw are conditional future experiments, not auto-launched.
Do not update paper scores until final configuration and evaluation protocol are
locked. Repeated seeds, full-video evaluation and cost profiling follow that review.

## Queue commands (repository root)

```bash
PYTHONPATH=scripts/thesis_experiment/src .venv/bin/python \
  scripts/thesis_experiment/tools/run_mam_softmax_fix_queue.py \
  --output Outputs_Thesis_Pilot20/mam_softmax_fix/<run-id> --dry-run

PYTHONPATH=scripts/thesis_experiment/src .venv/bin/python -u \
  scripts/thesis_experiment/tools/run_mam_softmax_fix_queue.py \
  --output Outputs_Thesis_Pilot20/mam_softmax_fix/<run-id> --gpus 0 1 2 3
```

Queue/GPU/worker locks prevent duplicate jobs within this workflow. GPU process
occupancy is checked before dispatch; other users' processes are never stopped.
Failures stop new dispatch and preserve outputs. Completed training is reused;
interrupted training resumes from last.pt. An interrupted heatmap-cache build
requires a fresh queue directory rather than accepting a partial cache. Read
`queue_status.json`, per-job logs, `diagnosis/selection.json`, `evaluation/`, and
`paired_comparisons.json`. Training artifacts use the unique `mam_softmax_fix`
stage under `Outputs_Thesis_Pilot20/runs/ntu60_cs/`.
