# NTU JTS Series

The Joint-aware Timestep Scheduling (JTS) series starts from the frozen
experimental contract of T0 and isolates SNN-internal time from RGB-video time.
JTS1-JTS5 receive one current RGB frame; `input_strategy: repeat` expands that
frame inside the model. Thus performance differences cannot be attributed to
additional historical video frames.

All formal screening runs use S010, seed 42, 20 epochs, disabled augmentation,
and T0's four-frame-eligible current-target population.

| ID | Stage steps | Definition |
|---|---|---|
| JTS0 | 1-1-1-1 | canonical T0 reference |
| JTS1 | 4-4-4-4 | full repeated-current SNN timestep control |
| JTS2 | 4-3-2-1 | aggressive stage-wise shrinkage |
| JTS3 | 4-4-3-1 | retain extra Stage-2/3 computation |
| JTS4 | 4-4-4-1 | retain four steps through Stage 3 |
| JTS5 | 2-2-2-2 | uniform low-budget control |
| JTS6 | probe | frozen equal-capacity readouts on Stages 1-4 |

Formal training outputs live under:

```text
Outputs_New/ntu_rgbd/joint_timestep/clip4_256_20ep/jts{0..5}/seed_42/
```

List or dry-run one training job:

```bash
.venv/bin/python scripts/NTU_RGBD/train_jts_series.py --list
.venv/bin/python scripts/NTU_RGBD/train_jts_series.py \
  --experiment jts3 --dry-run
```

Run the ranked four-GPU screen. The watcher retains and polls every child
process so completed training jobs are reaped rather than left as zombies. It
selects the highest mean wrist/hand/hand-tip/thumb PCKhn and runs JTS6 on GPU 0:

```bash
.venv/bin/python scripts/NTU_RGBD/train_jts_gpu_queue.py --dry-run
.venv/bin/python scripts/NTU_RGBD/train_jts_gpu_queue.py --launch
```

After a JTS repeated-current checkpoint completes, train JTS6 Stage probes:

```bash
.venv/bin/python scripts/NTU_RGBD/probe_jts_stages.py \
  --run Outputs_New/ntu_rgbd/joint_timestep/clip4_256_20ep/jts3/seed_42 \
  --output-dir Outputs_New/ntu_rgbd/joint_timestep/stage_probes_20ep/jts6/seed_42
```

Every run and Stage probe reports overall PCKhn@0.5 and per-joint results. JTS6
also reports the mean left/right scores for wrist, hand, hand tip, thumb, and
their combined hand group. Stage probes freeze the source backbone and use the
same one-layer readout, optimizer, epoch budget, and heatmap resolution.
