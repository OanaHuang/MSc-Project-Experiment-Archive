# NTU JATC Pilot Series

The Joint-Adaptive Time Constant pilot is a gated seven-run series built on the
frozen NTU T0 coordinate predictions. Hyperparameters are selected on S010 train
videos and evaluated once on S010 validation videos.

| ID | Seed | Stage | Definition |
|---|---:|---|---|
| JP0 | 42 | offline | raw T0 reference |
| JP1 | 42 | offline | shared fixed decay sweep |
| JP2 | 42 | offline | independent fixed decay per joint |
| JP3 | 42 | offline | confidence-motion rule decay |
| JP4 | 42 | training | learned joint-adaptive coordinate LIF |
| JP4-S2 | 3407 | replication | JP4 replication |
| JP4-S3 | 2026 | replication | JP4 replication |

Export T0 train and validation sequences (including heatmap confidence):

```bash
.venv/bin/python scripts/NTU_RGBD/run_jatc_series.py --export --device cuda:0
```

Inspect the manifest and commands:

```bash
.venv/bin/python scripts/NTU_RGBD/run_jatc_series.py --list
.venv/bin/python scripts/NTU_RGBD/train_jatc_pilot_queue.py --dry-run
```

Launch the gated queue:

```bash
.venv/bin/python scripts/NTU_RGBD/train_jatc_pilot_queue.py --launch \
  --gpu 0 --replication-gpus 0 1
```

The queue stops after JP2, JP3, or JP4 when its pre-registered accuracy and
NAccE gate fails. Existing output directories are never overwritten.
