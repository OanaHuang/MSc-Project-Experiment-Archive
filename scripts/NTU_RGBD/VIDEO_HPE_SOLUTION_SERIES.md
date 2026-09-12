# NTU Video-HPE Solution Series

The V-series converts four high-level video human-pose ideas into controlled,
causal, four-frame SpikePose experiments. All runs use the same S010
`clip4_256` targets, seed 42, disabled augmentation, and 20 epochs.

| ID | Literature direction | Local hypothesis |
|---|---|---|
| V1 | DCPose (CVPR 2021), pose residual fusion and correction | a small learned residual from aligned history can correct the current pose without allowing history to dominate |
| V2 | TDMI (CVPR 2023), task-relevant temporal differences | joint heatmap differences provide a cleaner motion gate than raw historical feature aggregation |
| V3 | DSTA (CVPR 2024), decoupled space-time aggregation | fuse each joint through time first, then refine only along the NTU skeleton graph |
| V4 | SmoothNet (ECCV 2022), position and acceleration objectives | matching second-order heatmap motion suppresses jitter without forcing constant predictions |

FAMI-Pose (CVPR 2022) motivates the shared alignment base used by every V run:
unaligned supporting-frame features are spatially mismatched under fast motion,
so historical joint heatmaps are aligned to the current soft joint coordinates
before correction or fusion.

The implementations are causal adaptations rather than reproductions of the
papers. This is necessary because the local task is single-person NTU pose
estimation with a spiking backbone, while the cited methods primarily target
PoseTrack-style architectures or pose-sequence refinement.

Launch only after the existing CF13 run completes:

```bash
.venv/bin/python scripts/NTU_RGBD/train_video_hpe_queue.py --dry-run
.venv/bin/python scripts/NTU_RGBD/train_video_hpe_queue.py --watch
```

Primary references:

- Liu et al., *Deep Dual Consecutive Network for Human Pose Estimation*, CVPR 2021.
- Liu et al., *Temporal Feature Alignment and Mutual Information Maximization for Video-Based Human Pose Estimation*, CVPR 2022.
- Feng et al., *Mutual Information-Based Temporal Difference Learning for Human Pose Estimation in Video*, CVPR 2023.
- He and Yang, *Video-Based Human Pose Regression via Decoupled Space-Time Aggregation*, CVPR 2024.
- Zeng et al., *SmoothNet: A Plug-and-Play Network for Refining Human Poses in Videos*, ECCV 2022.
