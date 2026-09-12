# MPII official-split backbone experiments

This study is the reproducible source for the thesis Table 1 comparison.  It
uses the fixed `train.json` / `valid.json` membership published by the official
HRNet repository.  The validation population is 2,958 people.  It is distinct
from the earlier seed-42 90/10 MPII split and its checkpoints must not be reused.

## Prepare data without training

The original MPII images, the official HRNet JSON annotations and the legacy
metadata must already exist below `Datasets/MPII`.

```bash
spikepose-thesis prepare-mpii-official
spikepose-thesis plan --study mpii_official_backbones
```

The preparation command verifies the official annotation SHA-256 values before
writing `Datasets/MPII/metadata/official_hrnet/{train,val}.jsonl` and a manifest.

Required released checkpoints:

```text
Datasets/MPII/official_simplebaseline/pose_resnet_50_256x256.pth.tar
Datasets/MPII/official_simplebaseline/pose_hrnet_w32_256x256.pth
```

Evaluate the released models without training:

```bash
spikepose-thesis evaluate-official-mpii \
  --experiment mpii_official_resnet50 --device cuda:0
spikepose-thesis evaluate-official-mpii \
  --experiment mpii_official_hrnet_w32 --device cuda:0
```

Both runs use the official affine crop, quarter-pixel decoder, flip test,
heatmap shift and PCKh@0.5.  ResNet-50 retains the official BGR convention;
HRNet-W32 retains RGB.  The output records both the published PCKh and the
locally reproduced value, along with checkpoint hashes.

`mpii_official_spikepose_ann`, `mpii_official_spikformer`,
`mpii_official_spikepose_u1` and `mpii_official_spikepose_u2` are
fresh-training configs and must not initialize
from the earlier random-split checkpoints.  No training is started by the data
preparation or planning commands.

`mpii_official_spikeyolo` is named SpikeYOLO in the paper and is configured for
scratch training.  Layers 0--25 provide the original 23M T1-D4 spatial network;
the detection layer is removed.  Layer 19 supplies the 128-channel stride-8 P3
feature to the same `linear_heatmap` head used by SpikePose: a 1x1 projection
to 16 joints followed by bilinear resize from 32x32 to 64x64.  The experiment
remains blocked until the external SpikeYOLO adapter has a reviewed standalone
implementation; the released detection checkpoint is not loaded.

`mpii_official_spikformer` is named Spikformer-T1 in the paper.  It uses the
original eight-block, 384-channel SPS/SSA backbone with one simulation step,
removes classification pooling/head layers, and attaches the same 16-joint
`linear_heatmap` head.  It is trained from scratch under the identical official
MPII protocol; no ImageNet accuracy or unofficial checkpoint is reused.
