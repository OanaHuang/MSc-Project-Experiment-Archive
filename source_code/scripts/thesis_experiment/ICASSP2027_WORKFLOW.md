# ICASSP final-paper experiment workflow

This is the frozen 16-joint baseline roster for the revised paper story. NTU-18
is now the mainline direction, but its authoritative mapping has not yet been
defined. Do not launch these NTU jobs as final NTU-18 training. The retained
results remain useful for reproducibility and warm-start comparisons.

## Fixed rules

- Use local development and `mitkof` only. Menorca results are excluded.
- Pilot20 and Confirm140 write to different output roots.
- Confirm140 starts independently. It never resumes or initializes from a
  Pilot checkpoint.
- MPII ANN-S3, S3-U1 and S3-U2 use three paper seeds in both phases.
- NTU main methods use seed 42 in Pilot20 and three seeds in Confirm140.
- NTU ablations use seed 42 only.
- SimpleBaseline-R50 and HRNet-W32 start from their official MPII checkpoints
  and are fine-tuned directly on NTU; they are not retrained on MPII.
- The original V1U1, V1U2 and V2U2 controls remain no-retraining evaluation
  jobs.  The Table-2 factor matrix is a separate set of six independently
  trained, single-seed NTU-full controls with matched target-frame eligibility.

## Dependency waves

| Phase | Wave 1 | Wave 2 | Wave 3 |
|---|---|---|---|
| Pilot20 | MPII ANN/S3-U1/S3-U2 | frame-wise SpikePose, R50, W32 | full MAM-v2 and three ablations |
| Confirm140 | independent MPII ANN/S3-U1/S3-U2 | independent frame-wise SpikePose, R50, W32 | full MAM-v2 and three ablations |
| Factor Pilot20 | six V/U controls, seed 42, NTU-full | -- | -- |
| Factor Confirm140 | six V/U controls, seed 42, NTU-full | -- | -- |
| Pilot eval | V1U1/V1U2/V2U2 and three filters | -- | -- |
| Confirm eval | V1U1/V1U2/V2U2 and three filters | -- | -- |

The three MAM-v2 ablations are:

- no alignment: no history warp;
- fixed leak: a learned time-invariant per-joint leak, without the dynamic
  motion-conditioned gate;
- no residual fusion: output the memory state directly instead of
  `current + gamma * (state - current)`. Motion residual-offset correction and
  the adaptive leak remain enabled.

## Required official checkpoints

The queue checks these files before it schedules the CNN baselines:

```text
Datasets/MPII/official_simplebaseline/pose_resnet_50_256x256.pth.tar
Datasets/MPII/official_simplebaseline/pose_hrnet_w32_256x256.pth
```

Each run records the file SHA-256 and the aligned tensor-state fingerprint in
`lineage.json`. Loading is strict; missing or incompatible tensors stop the
run.

## Mitkof commands

Run each dry-run first. A dry-run changes no experiment output.

```bash
cd /extra2/yunhao/MSc_Project/scripts/thesis_experiment
source ../../.venv/bin/activate
export PYTHONPATH=src

python -m spikepose_thesis validate
python tools/run_icassp2027_queue.py --phase pilot20 --dry-run
python tools/run_icassp2027_queue.py --phase confirm140 --dry-run
python tools/run_icassp2027_queue.py --phase pilot-eval --dry-run
python tools/run_icassp2027_queue.py --phase confirm-eval --dry-run
python tools/run_icassp2027_queue.py --phase pilot20-factors --dry-run
python tools/run_icassp2027_queue.py --phase confirm140-factors --dry-run
python tools/report_icassp2027_status.py --phase pilot20
```

Start a phase by removing `--dry-run`, for example:

```bash
python tools/run_icassp2027_queue.py --phase pilot20 --gpus 0 1 2 3
```

Recommended order:

1. `pilot20`
2. inspect Pilot validation results and freeze the paper design
3. `pilot-eval`
4. `confirm140`
5. `confirm-eval`

The factor controls may be launched independently after their matching MPII
S3-U2 source is complete.  Every condition follows one explicit continuous
update schedule and uses last-step readout:

| Condition | Physical frames | Continuous update schedule |
|---|---:|---|
| V1U1 | 1 | `[0]` |
| V1U2 | 1 | `[0, 0]` |
| V1U4 | 1 | `[0, 0, 0, 0]` |
| V2U2 | 2 | `[0, 1]` |
| V2U4 | 2 | `[0, 0, 1, 1]` |
| V4U4 | 4 | `[0, 1, 2, 3]` |

All six require three available history frames, so they train and validate on
the same supervised current-frame population.  Microbatch and accumulation
settings retain 128 supervised targets per optimizer update.

The queue uses a lock, writes an atomic `queue_status.json`, skips completed
runs, resumes interrupted training from `last.pt`, and rechecks dependencies
between waves. Existing full-NTU Pilot results for the frame model, full MAM,
no-alignment and fixed-leak variants keep their historical experiment IDs and
are detected rather than retrained.

## Canonical studies

- `icassp2027_pilot20`
- `icassp2027_confirm140`
- `icassp2027_pilot_eval`
- `icassp2027_confirm_eval`
- `ntu25_preview8` (retained auxiliary native-25 study)

The training studies contain sixteen experiment definitions each. The eval
studies contain six zero-epoch definitions each.
