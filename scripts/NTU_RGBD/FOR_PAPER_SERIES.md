# NTU RGB+D FP series for the paper

The `fp0`--`fp6` identifiers follow the existing lower-case series convention
(`t0`, `cf10`, `v1`). They are stable paper aliases: later experiments should
not silently change their parent configuration.

| ID | Method name | Controlled question |
|---|---|---|
| FP0 | CurrentFrame | Frame-wise SpikePose reference |
| FP1 | RepeatedCurrent | Does four-step compute alone help? |
| FP2 | FourFrameMean | Does unparameterized history help? |
| FP3 | JointTemporal | Do learned joint-wise temporal weights help? |
| FP4 | JointAlignedTemporal | Does alignment help beyond FP3? |
| FP5 | AlignedConfidenceGate | Does confidence gating help beyond FP4? |
| FP6 | CausalPoseResidualCorrection | Is current-frame-preserving residual correction safer? |

Every run predicts the same current-frame population (`minimum_temporal_history=3`),
uses the same clip cache and disables augmentation. FP0 still consumes one image;
eligibility filtering, rather than repeated input, keeps its evaluation population
matched to the four-frame methods.

Training protocol:

- `screen`: all methods, seed 42, 20 epochs. Debugging and method triage only.
- `main`: all methods, seeds 42/2071461405/365198782, 120 epochs. These are the
  runs used for mean and standard deviation comparisons.
- `extension`: all methods, seed 42, 140 epochs. This is a convergence diagnostic,
  not a substitute for the three-seed 120-epoch result.

Examples:

```bash
python scripts/NTU_RGBD/train_for_paper_series.py --list
python scripts/NTU_RGBD/train_for_paper_series.py --experiment fp0 --epochs 20 --dry-run
python scripts/NTU_RGBD/train_for_paper_queue.py --phase main --gpus 0 1 2 3 --dry-run
python scripts/NTU_RGBD/train_for_paper_queue.py --phase main --gpus 0 1 2 3 --launch
```

The current implementation treats the four video frames as the model's four
state-update steps. It must not be described as independently factorized video
time and SNN simulation time. A genuinely separated-time experiment requires a
new model implementation and a new FP identifier.
