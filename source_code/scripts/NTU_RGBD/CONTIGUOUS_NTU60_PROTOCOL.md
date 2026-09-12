# Contiguous NTU60 video-HPE protocol

The complete NTU60 RGB extraction uses two deterministic, non-overlapping
16-frame clips per original video. Original frame numbers are retained so RGB
frames and Kinect skeleton rows stay aligned.

Build portable Cross-Subject metadata after extraction:

```bash
.venv/bin/python scripts/NTU_RGBD/prepare_contiguous_metadata.py \
  --frame-root Datasets/NTU_RGBD/frames \
  --skeleton-root Datasets/NTU_RGBD/skeletons \
  --output-dir Datasets/NTU_RGBD/metadata/contiguous_xsub
```

The official NTU60 Cross-Subject test subjects are never used for training or
validation. Validation performers are deterministically held out from the
official training performers. `official_train_split.csv` is also emitted for a
final train-on-all-official-training-subjects run after hyperparameters are
frozen.

Train a four-frame causal video model on the sampled clips:

```bash
.venv/bin/python scripts/NTU_RGBD/train.py \
  --experiment v3 --category ntu60_contiguous \
  --batch-name xsub_2x16_t4 --seed 42 \
  --extracted-frames-dir Datasets/NTU_RGBD/frames \
  --contiguous-clip-subdir contiguous_2x16 \
  --train-metadata Datasets/NTU_RGBD/metadata/contiguous_xsub/train_split.csv \
  --validation-metadata Datasets/NTU_RGBD/metadata/contiguous_xsub/val_split.csv \
  --frame-stride 1 --temporal-frame-gap 1 \
  --minimum-temporal-history 3 --minimum-visible-joints 4
```

Dataset guarantees for this route:

- temporal histories never cross a `clip_XX` boundary;
- one square person crop is shared across all frames of a temporal sample;
- all clip frames share the same sampled geometric augmentation;
- target frames with fewer than the configured number of tracked joints are
  excluded;
- metadata stores project-relative skeleton paths and is portable between the
  workstation and mitkof;
- validation and test iteration are deterministic.

The extracted frames are a deterministic dataset subset, not stochastic epoch
sampling. Training still shuffles the resulting valid temporal windows each
epoch. Any paper must report the two clips per video, clip length 16, causal
window length, and official protocol used.
