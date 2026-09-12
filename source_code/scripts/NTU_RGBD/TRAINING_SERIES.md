# NTU RGB+D Training Series

All formal screening runs use S010, seed 42, and 20 epochs.

The external single-frame Pose-ResNet/HRNet comparisons are specified in
`NHB_SERIES.md` and use the same FP0-eligible sample population.

| Series | ID | Definition | Status | Legacy output |
|---|---|---|---|---|
| CF | CF0-Single | current-frame baseline | complete | `t_series/clip4_256_20ep/t0` |
| CF | CF1-Repeat2 | two repeated current frames | complete | `t_series/clip4_256_20ep/t1` |
| CF | CF2-JointTemporal | four-frame joint-wise aggregation | complete | `t_series/clip4_256_20ep/t2` |
| CF | CF3-SpaceTime | temporal aggregation plus skeleton refinement | complete | `t_series/clip4_256_20ep/t3` |
| CF | CF4-Confidence | confidence-gated aggregation | complete | `t_series/clip4_256_20ep/t4` |
| CF | CF5-Motion | motion-adaptive aggregation | ready | `cross_frame/clip4_256_20ep/t5` |
| CF | CF6-Repeat4 | four repeated current frames | ready | `cross_frame/clip4_256_20ep/cf6` |
| CF | CF7-Reverse | reversed temporal-order control | ready | `cross_frame/clip4_256_20ep/cf7` |
| CF | CF8-Shuffle | shuffled-history identity control | ready | `cross_frame/clip4_256_20ep/cf8` |
| CF | CF9-HistoryOnly | history-only forecast | queued | `cross_frame/clip4_256_20ep/cf9` |
| CF | CF10-Align | joint-wise feature alignment | queued | `cross_frame/clip4_256_20ep/cf10` |
| CF | CF11-AlignForecast | alignment plus forecast supervision | queued | `cross_frame/clip4_256_20ep/cf11` |
| CF | CF12-AlignMotion | alignment plus motion gate | queued | `cross_frame/clip4_256_20ep/cf12` |
| CF | CF13-AlignConfidence | alignment plus confidence gate | queued | `cross_frame/clip4_256_20ep/cf13` |
| T | T-4444 | stage timesteps 4-4-4-4 | complete | `t_ssnn/clip4_256_20ep/t_ssnn1` |
| T | T-4321 | stage timesteps 4-3-2-1 | complete | `t_ssnn/clip4_256_20ep/t_ssnn2` |
| T | T-4321-Aux | 4-3-2-1 plus auxiliary heads | complete | `t_ssnn/clip4_256_20ep/t_ssnn3` |
| JTS | JTS0-T0Reference | canonical T0 current-frame reference | ready | `joint_timestep/clip4_256_20ep/jts0` |
| JTS | JTS1-Repeat4 | repeated-current 4-4-4-4 compute control | ready | `joint_timestep/clip4_256_20ep/jts1` |
| JTS | JTS2-Shrink4321 | repeated-current stage timestep shrinkage | ready | `joint_timestep/clip4_256_20ep/jts2` |
| JTS | JTS3-Shrink4431 | preserve extra Stage-2/3 timesteps | ready | `joint_timestep/clip4_256_20ep/jts3` |
| JTS | JTS4-Shrink4441 | preserve four timesteps through Stage 3 | ready | `joint_timestep/clip4_256_20ep/jts4` |
| JTS | JTS5-Uniform2222 | repeated-current uniform low-budget control | ready | `joint_timestep/clip4_256_20ep/jts5` |
| JTS | JTS6-StageProbe | frozen equal-capacity Stage 1-4 readouts | conditional | `joint_timestep/stage_probes_20ep/jts6` |
| F | F1 | one input frame | queued | `frame_count/clip4_256_20ep/f1` |
| F | F2 | two adjacent input frames | queued | `frame_count/clip4_256_20ep/f2` |
| F | F3 | three adjacent input frames | queued | `frame_count/clip4_256_20ep/f3` |
| F | F4 | four adjacent input frames | queued | `frame_count/clip4_256_20ep/f4` |
| FS | FS1 | F1 plus 32-frame SmoothNet coordinate refinement | ready | `f_smoothnet/window32_50ep/fs1` |
| FS | FS2 | F2 plus 32-frame SmoothNet coordinate refinement | ready | `f_smoothnet/window32_50ep/fs2` |
| FS | FS3 | F3 plus 32-frame SmoothNet coordinate refinement | ready | `f_smoothnet/window32_50ep/fs3` |
| FS | FS4 | F4 plus 32-frame SmoothNet coordinate refinement | ready | `f_smoothnet/window32_50ep/fs4` |
| SN | SN0 | frozen T0 raw coordinate reference | complete | `sn_series/training_free_v1/sn0_raw` |
| SN | SN1 | T0 plus Savitzky-Golay W7/P2 | complete | `sn_series/training_free_v1/sn1_savgol_w7_p2` |
| SN | SN2 | T0 plus One Euro Filter | complete | `sn_series/training_free_v1/sn2_one_euro` |
| G | G1 | two frames, gap 1 | queued (corrected shared targets) | `frame_gap/clip6_256_20ep/g1` |
| G | G2 | two frames, gap 2 | queued (corrected shared targets) | `frame_gap/clip6_256_20ep/g2` |
| G | G3 | two frames, gap 3 | queued (corrected shared targets) | `frame_gap/clip6_256_20ep/g3` |
| G | G5 | two frames, gap 5 | queued (corrected shared targets) | `frame_gap/clip6_256_20ep/g5` |
| Control | R2 | repeated-current two-step reference | queued | `frame_gap_control/clip4_256_20ep/e0` |
| V | V1 | causal aligned pose-residual correction (DCPose) | queued after CF13 | `video_hpe_solution/clip4_256_20ep/v1` |
| V | V2 | joint temporal-difference gate (TDMI) | queued after CF13 | `video_hpe_solution/clip4_256_20ep/v2` |
| V | V3 | decoupled joint space-time fusion (DSTA) | queued after CF13 | `video_hpe_solution/clip4_256_20ep/v3` |
| V | V4 | acceleration-supervised aligned fusion (SmoothNet) | queued after CF13 | `video_hpe_solution/clip4_256_20ep/v4` |

Launch the first four-GPU CF batch:

```bash
.venv/bin/python scripts/NTU_RGBD/train_cf_gpu_batch.py --dry-run --epochs 20
.venv/bin/python scripts/NTU_RGBD/train_cf_gpu_batch.py --launch --epochs 20
```

Prepare the shared six-frame G source and launch the corrected G plus CF9-CF13 queue:

```bash
.venv/bin/python scripts/NTU_RGBD/prepare_clip4_256.py \
  --clip-length 6 --clip-start-gap 16 \
  --output-root Datasets/NTU_RGBD/frames/S010/clip6_256 \
  --metadata Datasets/NTU_RGBD/metadata/s010/train_split.csv \
  --metadata Datasets/NTU_RGBD/metadata/s010/val_split.csv
.venv/bin/python scripts/NTU_RGBD/train_ntu_followup_queue.py --dry-run
.venv/bin/python scripts/NTU_RGBD/train_ntu_followup_queue.py --launch
```
